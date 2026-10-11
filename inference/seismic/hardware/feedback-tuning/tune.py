"""Generic, budgeted search over finite opaque coordinates and an external observer.

No kernel names, shapes, source text, or performance rules enter the search.
The worker owns legal candidate formation, exact artifact reuse, and execution.
"""
import argparse
import itertools
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path


class Worker:
    def __init__(self, executable, fixture):
        self.process = subprocess.Popen([str(executable), fixture], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.metadata = self.receive()

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"observer exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def observe(self, point, trials, target_ms):
        request = dict(point=list(point), trials=trials, target_ms=float(target_ms))
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        # Requests are synchronous: every submitted GPU command has completed.
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


class Campaign:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.rng = random.Random(args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = args.output.open("w")
        self.worker = None
        self.observed = {}
        self.links = []
        self.costs = []
        self.new_artifacts = 0
        self.compile_ms = 0.0
        self.observations = 0
        self.next_progress = 0.0

    def elapsed(self):
        return time.monotonic() - self.started

    def log(self, kind, **data):
        self.stream.write(json.dumps(dict(kind=kind, elapsed_s=self.elapsed(), **data)) + "\n")
        self.stream.flush()

    def random_point(self):
        return tuple(self.rng.randrange(n) for n in self.axes)

    def evaluate(self, point, reason, *, fresh=False, trials=None, target_ms=None):
        if point in self.observed and not fresh:
            return self.observed[point]
        before = time.monotonic()
        trials = trials or self.args.trials
        target_ms = target_ms or self.args.target_ms
        result = self.worker.observe(point, trials, target_ms)
        samples = result["samples_ms"]
        record = dict(point=point, score=statistics.mean(samples),
                      sem=statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else 0.0,
                      samples=samples)
        if not fresh:
            self.observed[point] = record
        self.observations += 1
        self.costs.append(time.monotonic() - before)
        self.compile_ms += result["compile_ms"]
        self.new_artifacts += result["new_artifacts"]
        self.log("observation", reason=reason, fresh=fresh, trials=trials, target_ms=target_ms, **result)
        if self.elapsed() >= self.next_progress:
            best = min(self.observed.values(), key=lambda x: x["score"], default=record)
            print(json.dumps(dict(progress=reason, elapsed_s=round(self.elapsed(), 1),
                                  points=len(self.observed), best_ms=best["score"])), flush=True)
            self.next_progress = self.elapsed() + 10
        return record

    def population(self):
        return sorted(self.observed.values(), key=lambda x: x["score"])[:8]

    def room(self, deadline, count=1):
        # Empirical admission allowance, not a hard bound on driver compilation.
        allowance = max(self.costs[-32:] or [0.2]) * 1.5 + 0.1
        return self.elapsed() + count * allowance < deadline

    def mutation(self, point):
        changed = list(point)
        axis = self.rng.randrange(len(self.axes))
        changed[axis] = (changed[axis] + 1 + self.rng.randrange(self.axes[axis] - 1)) % self.axes[axis]
        return tuple(changed)

    def recombine(self):
        population = self.population()
        receiver = population[0] if self.rng.random() < 0.6 else self.rng.choice(population)
        # Poorer legal implementations remain available as donors.
        pool = list(self.observed.values()) if self.rng.random() < 0.25 else population
        donor = self.rng.choice(pool)
        differences = {i for i, (a, b) in enumerate(zip(receiver["point"], donor["point"])) if a != b}
        groups = [g & differences for g in self.links if g & differences]
        if groups and self.rng.random() < 0.7:
            group = self.rng.choice(groups)
        else:
            group = {i for i in differences if self.rng.random() < 0.5}
        child = tuple(donor["point"][i] if i in group else v for i, v in enumerate(receiver["point"]))
        if child == receiver["point"] or child in self.observed:
            child = self.mutation(child)
        return child

    def dissect(self, deadline):
        a = self.population()[0]
        possible = [b for b in self.observed.values()
                    if sum(x != y for x, y in zip(a["point"], b["point"])) >= 2]
        if not possible or not self.room(deadline, 2):
            return False
        # Compare different derivations; the search does not see their meaning.
        b = self.rng.choice(possible)
        different = [i for i, (x, y) in enumerate(zip(a["point"], b["point"])) if x != y]
        self.rng.shuffle(different)
        left = set(different[:len(different) // 2])
        right = set(different) - left
        def replace(group):
            return tuple(b["point"][i] if i in group else v for i, v in enumerate(a["point"]))
        p, q = replace(left), replace(right)
        if p in self.observed and q in self.observed:
            return False
        x = self.evaluate(p, "dissection_left")
        if not self.room(deadline):
            return True
        y = self.evaluate(q, "dissection_right")
        interaction = b["score"] - x["score"] - y["score"] + a["score"]
        noise = math.sqrt(sum(r["sem"] ** 2 for r in [a, b, x, y]))
        # A proposal heuristic only: correlated batches and drifting old anchors
        # do not turn this diagnostic into a calibrated independence certificate.
        linked = abs(interaction) > max(3 * noise, a["score"] * 0.01)
        if linked:
            group = frozenset(different)
            if group not in self.links:
                self.links.append(group)
                self.links = self.links[-32:]
        self.log("interaction", a=a["point"], b=b["point"], left=sorted(left), right=sorted(right),
                 delta_ms=interaction, diagnostic_noise_ms=noise, linked_for_proposals=linked)
        return True

    def finalize(self, deadline):
        ranked = sorted(self.observed.values(), key=lambda r: r["score"])
        finalists = [r["point"] for r in ranked[:self.args.finalists]]
        baseline = tuple(0 for _ in self.axes)
        if baseline not in finalists:
            finalists.append(baseline)
        samples = {p: [] for p in finalists}
        # Randomized fresh observations, same endpoint, independent of nomination.
        completed_rounds = 0
        for _ in range(2):
            if not self.room(deadline, len(finalists)):
                break
            order = list(finalists)
            self.rng.shuffle(order)
            for point in order:
                r = self.evaluate(point, "finalist_validation", fresh=True, trials=10, target_ms=8)
                samples[point].extend(r["samples"])
            completed_rounds += 1
        if completed_rounds:
            selected = min(finalists, key=lambda p: statistics.mean(samples[p]))
            score = statistics.mean(samples[selected])
            baseline_score = statistics.mean(samples[baseline])
        else:
            # No fresh fair comparison: retain the baseline, not a noisy winner.
            selected = baseline
            score = baseline_score = self.observed[baseline]["score"]
        self.log("selection", point=selected, score_ms=score, baseline_ms=baseline_score,
                 validation_rounds=completed_rounds,
                 finalists=[dict(point=p, samples_ms=samples[p]) for p in finalists])
        return selected, score, baseline_score, completed_rounds

    def run(self):
        try:
            self.worker = Worker(self.args.worker.resolve(), self.args.fixture)
            self.axes = self.worker.metadata["axes"]
            self.total_points = math.prod(self.axes)
            self.log("environment", algorithm=self.args.algorithm, seed=self.args.seed,
                     budget_s=self.args.budget, total_points=self.total_points,
                     **{k: v for k, v in self.worker.metadata.items() if k != "kind"})
            self.evaluate(tuple(0 for _ in self.axes), "baseline")
            deadline = self.args.budget - min(10, self.args.budget / 4)
            if self.args.algorithm == "exhaustive":
                points = list(itertools.product(*(range(n) for n in self.axes)))
                self.rng.shuffle(points)
                for point in points:
                    if not self.room(deadline):
                        break
                    self.evaluate(point, "enumeration")
            else:
                for _ in range(3):
                    if self.room(deadline):
                        self.evaluate(self.random_point(), "seed")
                iteration = 0
                while len(self.observed) < self.total_points and self.room(deadline):
                    iteration += 1
                    if self.args.algorithm == "evolution" and iteration % 5 == 0 and self.dissect(deadline):
                        continue
                    if self.args.algorithm == "random" or self.rng.random() < 0.25:
                        point = self.random_point(); reason = "exploration"
                    else:
                        point = self.recombine(); reason = "crossover_or_mutation"
                    for _ in range(32):
                        if point not in self.observed:
                            break
                        point = self.random_point()
                    if point not in self.observed:
                        self.evaluate(point, reason)
            selected, score, baseline_score, rounds = self.finalize(self.args.budget)
            result = dict(fixture=self.args.fixture, algorithm=self.args.algorithm, seed=self.args.seed,
                          elapsed_s=self.elapsed(), within_budget=self.elapsed() <= self.args.budget,
                          unique_points=len(self.observed), total_points=self.total_points,
                          observations=self.observations, new_artifacts=self.new_artifacts,
                          compile_ms=self.compile_ms, learned_groups=len(self.links),
                          selected=list(selected), score_ms=score, baseline_ms=baseline_score,
                          speedup=baseline_score / score, validation_rounds=rounds)
            self.log("result", **{k: v for k, v in result.items() if k != "elapsed_s"})
            print(json.dumps(result), flush=True)
            return result
        finally:
            if self.worker is not None:
                self.worker.close()
            self.stream.close()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, default=Path("./metal-worker"))
    parser.add_argument("--fixture", required=True, help="Opaque workload name understood by the observer")
    parser.add_argument("--algorithm", choices=["evolution", "random", "exhaustive"], default="evolution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=60)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=6)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 5 or not 1 <= args.trials <= 32 or not 0 < args.target_ms <= 100 or not 1 <= args.finalists <= 64:
        parser.error("invalid budget or observation protocol")
    return args


if __name__ == "__main__":
    Campaign(arguments()).run()
