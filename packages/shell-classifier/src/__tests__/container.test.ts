import { describe, test, expect } from 'bun:test'
import { isContainerForbidden } from '../tools/container'

const a = (s: string) => s.split(/\s+/).filter(Boolean)

const PUSH_REASON = 'Pushing images mutates remote registries. Use `docker build` and `docker tag` for local image management.'
const LOGIN_REASON = 'Docker login mutates remote authentication state. Use local image operations instead.'
const LOGOUT_REASON = 'Docker logout mutates remote authentication state.'

const PRUNE_REASON = 'Prune can remove broad sets of images, volumes, and caches across workflows. Use targeted cleanup like `docker rm <id>` or `docker rmi <id>` instead.'
const COMPOSE_DOWN_REASON = 'This can delete persistent volumes/images and cause data loss. Use `docker compose stop` or `docker compose down` without destructive flags.'
const PRIVILEGED_REASON = 'Privileged mode removes key isolation and increases host impact risk. Run unprivileged and grant only minimal capabilities if required.'
const HOST_NAMESPACE_REASON = 'Host namespace sharing weakens containment and broadens blast radius. Use default namespaces and explicit port mappings instead.'
const CAP_SECURITY_REASON = 'Disabling sandbox controls or adding broad caps enables host-like behavior. Keep default security profiles and least-privilege capabilities.'
const SENSITIVE_MOUNT_REASON = 'Mounting sensitive host paths can expose secrets/system files to container code. Mount only required project directories.'

describe('isContainerForbidden', () => {
  describe('allowed baseline', () => {
    const allowed = [
      'build .',
      'pull alpine',
      'pull myimage',
      'search nginx',
      'run alpine',
      'exec -it c sh',
      'compose up',
      'compose down',
      'rm abc123',
      'rmi alpine',
    ]

    for (const cmd of allowed) {
      test(`docker ${cmd} => null`, () => {
        expect(isContainerForbidden('docker', a(cmd))).toBeNull()
      })
    }
  })

  describe('remote mutating subcommands blocked', () => {
    test('docker push myimage => push reason', () => {
      expect(isContainerForbidden('docker', a('push myimage'))).toBe(PUSH_REASON)
    })

    test('docker login => login reason', () => {
      expect(isContainerForbidden('docker', a('login'))).toBe(LOGIN_REASON)
    })

    test('docker logout => logout reason', () => {
      expect(isContainerForbidden('docker', a('logout'))).toBe(LOGOUT_REASON)
    })

    test('podman push myimage => push reason', () => {
      expect(isContainerForbidden('podman', a('push myimage'))).toBe(PUSH_REASON)
    })

    test('nerdctl push myimage => push reason', () => {
      expect(isContainerForbidden('nerdctl', a('push myimage'))).toBe(PUSH_REASON)
    })
  })

  describe('prune blocked', () => {
    const dockerCases = [
      'system prune -a',
      'container prune',
      'image prune -a',
      'volume prune',
      'network prune',
      'builder prune',
    ]

    for (const cmd of dockerCases) {
      test(`docker ${cmd} => prune reason`, () => {
        expect(isContainerForbidden('docker', a(cmd))).toBe(PRUNE_REASON)
      })
    }

    test('podman system prune -a => prune reason', () => {
      expect(isContainerForbidden('podman', a('system prune -a'))).toBe(PRUNE_REASON)
    })

    test('nerdctl image prune -a => prune reason', () => {
      expect(isContainerForbidden('nerdctl', a('image prune -a'))).toBe(PRUNE_REASON)
    })
  })

  describe('compose down destructive blocked', () => {
    test('docker compose down -v', () => {
      expect(isContainerForbidden('docker', a('compose down -v'))).toBe(COMPOSE_DOWN_REASON)
    })

    test('docker compose down --volumes', () => {
      expect(isContainerForbidden('docker', a('compose down --volumes'))).toBe(COMPOSE_DOWN_REASON)
    })

    test('docker compose down --rmi all', () => {
      expect(isContainerForbidden('docker', a('compose down --rmi all'))).toBe(COMPOSE_DOWN_REASON)
    })

    test('docker compose down --remove-orphans', () => {
      expect(isContainerForbidden('docker', a('compose down --remove-orphans'))).toBe(COMPOSE_DOWN_REASON)
    })

    test('docker compose up -v is not blocked by compose-down rule', () => {
      expect(isContainerForbidden('docker', a('compose up -v'))).toBeNull()
    })
  })

  describe('privileged / namespace flags blocked (run-like only)', () => {
    test('docker run --privileged alpine', () => {
      expect(isContainerForbidden('docker', a('run --privileged alpine'))).toBe(PRIVILEGED_REASON)
    })

    test('docker run --pid=host alpine', () => {
      expect(isContainerForbidden('docker', a('run --pid=host alpine'))).toBe(HOST_NAMESPACE_REASON)
    })

    test('docker run --network=host alpine', () => {
      expect(isContainerForbidden('docker', a('run --network=host alpine'))).toBe(HOST_NAMESPACE_REASON)
    })

    test('docker run --net=host alpine', () => {
      expect(isContainerForbidden('docker', a('run --net=host alpine'))).toBe(HOST_NAMESPACE_REASON)
    })

    test('docker compose run --privileged svc', () => {
      expect(isContainerForbidden('docker', a('compose run --privileged svc'))).toBe(PRIVILEGED_REASON)
    })

    test('docker compose up --network=host', () => {
      expect(isContainerForbidden('docker', a('compose up --network=host'))).toBe(HOST_NAMESPACE_REASON)
    })

    test('docker build --network=host . is allowed (run-like gating)', () => {
      expect(isContainerForbidden('docker', a('build --network=host .'))).toBeNull()
    })
  })

  describe('cap/security flags blocked', () => {
    test('docker run --cap-add=ALL alpine', () => {
      expect(isContainerForbidden('docker', a('run --cap-add=ALL alpine'))).toBe(CAP_SECURITY_REASON)
    })

    test('docker run --cap-add=all alpine', () => {
      expect(isContainerForbidden('docker', a('run --cap-add=all alpine'))).toBe(CAP_SECURITY_REASON)
    })

    test('docker run --cap-add=SYS_ADMIN alpine', () => {
      expect(isContainerForbidden('docker', a('run --cap-add=SYS_ADMIN alpine'))).toBe(CAP_SECURITY_REASON)
    })

    test('docker run --security-opt seccomp=unconfined alpine', () => {
      expect(isContainerForbidden('docker', a('run --security-opt seccomp=unconfined alpine'))).toBe(CAP_SECURITY_REASON)
    })

    test('docker run --security-opt=apparmor=unconfined alpine', () => {
      expect(isContainerForbidden('docker', a('run --security-opt=apparmor=unconfined alpine'))).toBe(CAP_SECURITY_REASON)
    })
  })

  describe('sensitive mounts blocked', () => {
    const blocked = [
      'run -v /:/host alpine',
      'run -v /var/run/docker.sock:/var/run/docker.sock alpine',
      'run --volume /etc:/host/etc alpine',
      'run --mount type=bind,source=/root,target=/hostroot alpine',
      'run -v ~/.ssh:/root/.ssh alpine',
      'run -v ~/.aws:/root/.aws alpine',
      'run -v ~/.config/gcloud:/root/.config/gcloud alpine',
      'run -v ~/.azure:/root/.azure alpine',
    ]

    for (const cmd of blocked) {
      test(`docker ${cmd} => sensitive mount reason`, () => {
        expect(isContainerForbidden('docker', a(cmd))).toBe(SENSITIVE_MOUNT_REASON)
      })
    }

    test('docker run -v ./project:/workspace alpine => null', () => {
      expect(isContainerForbidden('docker', a('run -v ./project:/workspace alpine'))).toBeNull()
    })
  })

  describe('regression / parsing edge cases', () => {
    test('-v/tmp:/tmp inline form is parsed and allowed when not sensitive', () => {
      expect(isContainerForbidden('docker', a('run -v/tmp:/tmp alpine'))).toBeNull()
    })

    test('--volume=/etc:/x blocked', () => {
      expect(isContainerForbidden('docker', a('run --volume=/etc:/x alpine'))).toBe(SENSITIVE_MOUNT_REASON)
    })

    test('--mount=type=bind,src=/var/run/docker.sock,target=/sock blocked', () => {
      expect(isContainerForbidden('docker', a('run --mount=type=bind,src=/var/run/docker.sock,target=/sock alpine'))).toBe(SENSITIVE_MOUNT_REASON)
    })

    test('docker ps --security-opt seccomp=unconfined => null (run-like gating)', () => {
      expect(isContainerForbidden('docker', a('ps --security-opt seccomp=unconfined'))).toBeNull()
    })

    test('empty args => null', () => {
      expect(isContainerForbidden('docker', [])).toBeNull()
    })
  })
})