import { describe, expect, it } from 'bun:test'
import { isCloudCliForbidden } from '../tools/cloud-cli'

describe('cloud-cli tool policy (verb-prefix)', () => {
  describe('aws', () => {
    it('allows read and local config commands', () => {
      expect(isCloudCliForbidden('aws', ['ec2', 'describe-instances'])).toBeNull()
      expect(isCloudCliForbidden('aws', ['ec2', 'list-images'])).toBeNull()
      expect(isCloudCliForbidden('aws', ['sts', 'get-caller-identity'])).toBeNull()
      expect(isCloudCliForbidden('aws', ['configure', 'set', 'region', 'us-east-1'])).toBeNull()
    })

    it('forbids mutating verb-prefix actions', () => {
      expect(isCloudCliForbidden('aws', ['ec2', 'terminate-instances', '--instance-ids', 'i-1'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['ec2', 'run-instances', '--image-id', 'ami-1'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['ec2', 'start-instances', '--instance-ids', 'i-1'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['ec2', 'stop-instances', '--instance-ids', 'i-1'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['kms', 'schedule-key-deletion', '--key-id', 'abc'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['iam', 'attach-role-policy', '--role-name', 'r'])).not.toBeNull()
    })

    it('forbids s3 high-level mutating commands and allows ls', () => {
      expect(isCloudCliForbidden('aws', ['s3', 'cp', 'a', 's3://b'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'mv', 'a', 's3://b'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'rm', 's3://b/k'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'rb', 's3://b'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'sync', '.', 's3://b'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'mb', 's3://b'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3', 'ls', 's3://b'])).toBeNull()
    })

    it('applies verb-prefix checks to s3api', () => {
      expect(isCloudCliForbidden('aws', ['s3api', 'put-object', '--bucket', 'b', '--key', 'k'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3api', 'delete-object', '--bucket', 'b', '--key', 'k'])).not.toBeNull()
      expect(isCloudCliForbidden('aws', ['s3api', 'get-object', '--bucket', 'b', '--key', 'k', 'out'])).toBeNull()
    })
  })

  describe('gcloud', () => {
    it('allows read verbs and config/version/info', () => {
      expect(isCloudCliForbidden('gcloud', ['projects', 'list'])).toBeNull()
      expect(isCloudCliForbidden('gcloud', ['compute', 'instances', 'describe', 'vm-1'])).toBeNull()
      expect(isCloudCliForbidden('gcloud', ['config', 'set', 'project', 'my-dev'])).toBeNull()
      expect(isCloudCliForbidden('gcloud', ['version'])).toBeNull()
      expect(isCloudCliForbidden('gcloud', ['info'])).toBeNull()
    })

    it('handles auth policy', () => {
      expect(isCloudCliForbidden('gcloud', ['auth', 'list'])).toBeNull()
      expect(isCloudCliForbidden('gcloud', ['auth', 'revoke', 'me@example.com'])).not.toBeNull()
    })

    it('forbids mutating verbs', () => {
      expect(isCloudCliForbidden('gcloud', ['projects', 'delete', 'my-prod'])).not.toBeNull()
      expect(isCloudCliForbidden('gcloud', ['compute', 'instances', 'create', 'vm-1'])).not.toBeNull()
      expect(isCloudCliForbidden('gcloud', ['compute', 'instances', 'stop', 'vm-1'])).not.toBeNull()
      expect(isCloudCliForbidden('gcloud', ['run', 'deploy', 'svc'])).not.toBeNull()
      expect(isCloudCliForbidden('gcloud', ['dns', 'record-sets', 'import', 'zone'])).not.toBeNull()
      expect(isCloudCliForbidden('gcloud', ['projects', 'add-iam-policy-binding', 'p'])).not.toBeNull()
    })
  })

  describe('az', () => {
    it('allows read verbs and local commands', () => {
      expect(isCloudCliForbidden('az', ['account', 'show'])).toBeNull()
      expect(isCloudCliForbidden('az', ['group', 'list'])).toBeNull()
      expect(isCloudCliForbidden('az', ['config', 'set', 'defaults.group=rg'])).toBeNull()
      expect(isCloudCliForbidden('az', ['configure'])).toBeNull()
      expect(isCloudCliForbidden('az', ['init'])).toBeNull()
      expect(isCloudCliForbidden('az', ['version'])).toBeNull()
    })

    it('forbids mutating verbs', () => {
      expect(isCloudCliForbidden('az', ['group', 'create', '-n', 'rg', '-l', 'eastus'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['group', 'delete', '-n', 'rg'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['vm', 'start', '-g', 'rg', '-n', 'vm'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['vm', 'deallocate', '-g', 'rg', '-n', 'vm'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['resource', 'move', '--ids', 'x', '--destination-group', 'y'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['appservice', 'plan', 'update', '-g', 'rg', '-n', 'p'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['keyvault', 'purge', '-n', 'kv'])).not.toBeNull()
    })

    it('forbids deployment create with mode complete', () => {
      expect(isCloudCliForbidden('az', ['deployment', 'group', 'create', '--mode', 'complete'])).not.toBeNull()
      expect(isCloudCliForbidden('az', ['deployment', 'sub', 'create', '--mode=complete'])).not.toBeNull()
    })
  })

  it('returns null for unsupported cloud base', () => {
    expect(isCloudCliForbidden('doctl', ['compute', 'droplet', 'delete', 'x'])).toBeNull()
  })
})