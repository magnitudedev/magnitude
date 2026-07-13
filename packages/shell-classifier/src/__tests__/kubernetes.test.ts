import { describe, expect, test } from 'bun:test'
import { isHelmForbidden, isKubectlForbidden } from '../tools/kubernetes'

describe('kubernetes tool policy (read-allowed, write-forbidden)', () => {
  test('empty args return null', () => {
    expect(isKubectlForbidden([])).toBeNull()
    expect(isHelmForbidden([])).toBeNull()
  })

  describe('kubectl allowed reads', () => {
    test('core read commands', () => {
      expect(isKubectlForbidden(['get', 'pods'])).toBeNull()
      expect(isKubectlForbidden(['describe', 'pod', 'api'])).toBeNull()
      expect(isKubectlForbidden(['logs', 'api'])).toBeNull()
      expect(isKubectlForbidden(['top', 'pods'])).toBeNull()
      expect(isKubectlForbidden(['explain', 'deployment'])).toBeNull()
      expect(isKubectlForbidden(['diff', '-f', 'k8s.yaml'])).toBeNull()
    })

    test('read-allowed operational helpers', () => {
      expect(isKubectlForbidden(['exec', '-it', 'pod/api', '--', 'sh'])).toBeNull()
      expect(isKubectlForbidden(['port-forward', 'svc/api', '8080:80'])).toBeNull()
      expect(isKubectlForbidden(['proxy'])).toBeNull()
      expect(isKubectlForbidden(['cp', 'pod/api:/tmp/a', './a'])).toBeNull()
      expect(isKubectlForbidden(['config', 'view'])).toBeNull()
      expect(isKubectlForbidden(['plugin', 'list'])).toBeNull()
    })

    test('allowed compound read commands', () => {
      expect(isKubectlForbidden(['rollout', 'status', 'deploy/api'])).toBeNull()
      expect(isKubectlForbidden(['rollout', 'history', 'deploy/api'])).toBeNull()
      expect(isKubectlForbidden(['auth', 'can-i', 'get', 'pods'])).toBeNull()
      expect(isKubectlForbidden(['auth', 'whoami'])).toBeNull()
    })
  })

  describe('kubectl forbidden writes', () => {
    test('forbidden primary mutators', () => {
      expect(isKubectlForbidden(['apply', '-f', 'k8s.yaml'])).not.toBeNull()
      expect(isKubectlForbidden(['create', 'deployment', 'api', '--image=api'])).not.toBeNull()
      expect(isKubectlForbidden(['delete', 'pod', 'api'])).not.toBeNull()
      expect(isKubectlForbidden(['patch', 'deploy', 'api', '-p', '{}'])).not.toBeNull()
      expect(isKubectlForbidden(['scale', 'deploy', 'api', '--replicas=3'])).not.toBeNull()
      expect(isKubectlForbidden(['drain', 'node-1'])).not.toBeNull()
      expect(isKubectlForbidden(['taint', 'nodes', 'n1', 'k=v:NoSchedule'])).not.toBeNull()
      expect(isKubectlForbidden(['annotate', 'pod', 'api', 'x=y'])).not.toBeNull()
      expect(isKubectlForbidden(['label', 'pod', 'api', 'x=y'])).not.toBeNull()
      expect(isKubectlForbidden(['debug', 'pod/api'])).not.toBeNull()
    })

    test('forbidden compound mutators', () => {
      expect(isKubectlForbidden(['rollout', 'restart', 'deploy/api'])).not.toBeNull()
      expect(isKubectlForbidden(['rollout', 'undo', 'deploy/api'])).not.toBeNull()
      expect(isKubectlForbidden(['rollout', 'pause', 'deploy/api'])).not.toBeNull()
      expect(isKubectlForbidden(['rollout', 'resume', 'deploy/api'])).not.toBeNull()
      expect(isKubectlForbidden(['set', 'image', 'deploy/api', 'api=img:v2'])).not.toBeNull()
      expect(isKubectlForbidden(['auth', 'reconcile', '-f', 'rbac.yaml'])).not.toBeNull()
      expect(isKubectlForbidden(['certificate', 'approve', 'csr-1'])).not.toBeNull()
      expect(isKubectlForbidden(['certificate', 'deny', 'csr-1'])).not.toBeNull()
    })

    test('dangerous flag catches', () => {
      expect(isKubectlForbidden(['get', 'pods', '--force'])).not.toBeNull()
      expect(isKubectlForbidden(['get', 'pods', '--grace-period=0'])).not.toBeNull()
      expect(isKubectlForbidden(['get', 'pods', '--grace-period', '0'])).not.toBeNull()
      expect(isKubectlForbidden(['get', 'pods', '--all'])).not.toBeNull()
      expect(isKubectlForbidden(['get', 'pods', '-A'])).not.toBeNull()
      expect(isKubectlForbidden(['get', 'pods', '--all-namespaces'])).not.toBeNull()
    })
  })

  describe('helm allowed reads/local commands', () => {
    test('read-only or local-safe helm commands', () => {
      expect(isHelmForbidden(['list'])).toBeNull()
      expect(isHelmForbidden(['status', 'api'])).toBeNull()
      expect(isHelmForbidden(['get', 'values', 'api'])).toBeNull()
      expect(isHelmForbidden(['history', 'api'])).toBeNull()
      expect(isHelmForbidden(['search', 'repo', 'nginx'])).toBeNull()
      expect(isHelmForbidden(['pull', 'repo/chart'])).toBeNull()
      expect(isHelmForbidden(['repo', 'list'])).toBeNull()
      expect(isHelmForbidden(['completion', 'bash'])).toBeNull()
      expect(isHelmForbidden(['create', 'chart'])).toBeNull()
      expect(isHelmForbidden(['env'])).toBeNull()
      expect(isHelmForbidden(['lint', './chart'])).toBeNull()
      expect(isHelmForbidden(['package', './chart'])).toBeNull()
      expect(isHelmForbidden(['template', 'api', './chart'])).toBeNull()
      expect(isHelmForbidden(['show', 'values', 'repo/chart'])).toBeNull()
      expect(isHelmForbidden(['verify', './chart.tgz'])).toBeNull()
      expect(isHelmForbidden(['version'])).toBeNull()
    })
  })

  describe('helm forbidden writes', () => {
    test('forbidden primary mutators', () => {
      expect(isHelmForbidden(['install', 'api', './chart'])).not.toBeNull()
      expect(isHelmForbidden(['upgrade', 'api', './chart'])).not.toBeNull()
      expect(isHelmForbidden(['uninstall', 'api'])).not.toBeNull()
      expect(isHelmForbidden(['rollback', 'api', '1'])).not.toBeNull()
      expect(isHelmForbidden(['test', 'api'])).not.toBeNull()
    })

    test('forbidden compound mutators', () => {
      expect(isHelmForbidden(['push', './chart.tgz', 'oci://repo'])).not.toBeNull()
      expect(isHelmForbidden(['registry', 'login', 'repo.example.com'])).not.toBeNull()
      expect(isHelmForbidden(['registry', 'logout', 'repo.example.com'])).not.toBeNull()
      expect(isHelmForbidden(['repo', 'add', 'stable', 'https://example.com'])).not.toBeNull()
      expect(isHelmForbidden(['repo', 'remove', 'stable'])).not.toBeNull()
      expect(isHelmForbidden(['repo', 'update'])).not.toBeNull()
      expect(isHelmForbidden(['plugin', 'install', 'https://example.com/plugin'])).not.toBeNull()
      expect(isHelmForbidden(['plugin', 'uninstall', 'p'])).not.toBeNull()
      expect(isHelmForbidden(['plugin', 'update', 'p'])).not.toBeNull()
    })

    test('force flag forbidden on any helm command', () => {
      expect(isHelmForbidden(['status', 'api', '--force'])).not.toBeNull()
    })
  })

  describe('reason spot checks', () => {
    test('kubectl mutation reason mentions mutate and read alternatives', () => {
      const reason = isKubectlForbidden(['apply', '-f', 'k8s.yaml'])
      expect(reason).toContain('mutates')
      expect(reason).toContain('kubectl get')
    })

    test('helm mutation reason mentions read alternatives', () => {
      const reason = isHelmForbidden(['install', 'api', './chart'])
      expect(reason).toContain('helm')
      expect(reason).toContain('helm list')
    })
  })
})