import { describe, expect, test } from 'bun:test'
import { isIacForbidden } from '../tools/iac'

describe('isIacForbidden', () => {
  describe('terraform / terragrunt', () => {
    test('allows terraform plan', () => {
      expect(isIacForbidden('terraform', ['plan'])).toBeNull()
    })

    test('allows terraform fmt', () => {
      expect(isIacForbidden('terraform', ['fmt'])).toBeNull()
    })

    test('forbids terraform apply', () => {
      expect(isIacForbidden('terraform', ['apply'])).toContain('Applying infrastructure changes mutates remote state')
    })

    test('forbids terraform destroy', () => {
      expect(isIacForbidden('terraform', ['destroy', '-auto-approve'])).toContain('Destroy tears down')
    })

    test('forbids terraform import', () => {
      expect(isIacForbidden('terraform', ['import', 'aws_s3_bucket.logs', 'bucket-id'])).toContain('Import mutates Terraform state')
    })

    test('forbids terraform state push', () => {
      expect(isIacForbidden('terraform', ['state', 'push', 'terraform.tfstate'])).toContain('Direct state mutation')
    })

    test('forbids terraform state replace-provider', () => {
      expect(isIacForbidden('terraform', ['state', 'replace-provider', 'old', 'new'])).toContain('Direct state mutation')
    })

    test('forbids terraform workspace new', () => {
      expect(isIacForbidden('terraform', ['workspace', 'new', 'prod'])).toContain('Workspace mutations')
    })

    test('forbids terraform workspace delete', () => {
      expect(isIacForbidden('terraform', ['workspace', 'delete', 'prod'])).toContain('Workspace mutations')
    })

    test('forbids terraform workspace select', () => {
      expect(isIacForbidden('terraform', ['workspace', 'select', 'prod'])).toContain('Workspace mutations')
    })

    test('forbids terraform taint', () => {
      expect(isIacForbidden('terraform', ['taint', 'aws_instance.web'])).toContain('Taint/untaint mutates')
    })

    test('forbids terraform untaint', () => {
      expect(isIacForbidden('terraform', ['untaint', 'aws_instance.web'])).toContain('Taint/untaint mutates')
    })

    test('forbids terragrunt apply', () => {
      expect(isIacForbidden('terragrunt', ['apply'])).toContain('Applying infrastructure changes mutates remote state')
    })

    test('forbids terragrunt state rm', () => {
      expect(isIacForbidden('terragrunt', ['state', 'rm', 'module.db'])).toContain('Direct state mutation')
    })
  })

  describe('pulumi', () => {
    test('allows pulumi preview', () => {
      expect(isIacForbidden('pulumi', ['preview'])).toBeNull()
    })

    test('allows pulumi stack ls', () => {
      expect(isIacForbidden('pulumi', ['stack', 'ls'])).toBeNull()
    })

    test('forbids pulumi up', () => {
      expect(isIacForbidden('pulumi', ['up'])).toContain('Pulumi up deploys infrastructure changes')
    })

    test('forbids pulumi destroy', () => {
      expect(isIacForbidden('pulumi', ['destroy'])).toContain('Destroy removes stack resources')
    })

    test('forbids pulumi stack init', () => {
      expect(isIacForbidden('pulumi', ['stack', 'init', 'prod'])).toContain('Stack mutation commands')
    })

    test('forbids pulumi stack import', () => {
      expect(isIacForbidden('pulumi', ['stack', 'import', '--file', 'state.json'])).toContain('Stack mutation commands')
    })

    test('forbids pulumi stack rm', () => {
      expect(isIacForbidden('pulumi', ['stack', 'rm', 'prod'])).toContain('Stack mutation commands')
    })

    test('forbids pulumi state delete', () => {
      expect(isIacForbidden('pulumi', ['state', 'delete', 'urn:...'])).toContain('Direct state mutation')
    })

    test('forbids pulumi state unprotect', () => {
      expect(isIacForbidden('pulumi', ['state', 'unprotect', 'urn:...'])).toContain('Direct state mutation')
    })

    test('forbids pulumi cancel', () => {
      expect(isIacForbidden('pulumi', ['cancel'])).toContain('Cancel mutates in-progress deployment state')
    })
  })

  describe('sst', () => {
    test('allows sst diff', () => {
      expect(isIacForbidden('sst', ['diff'])).toBeNull()
    })

    test('forbids sst deploy', () => {
      expect(isIacForbidden('sst', ['deploy'])).toContain('Deploy mutates remote cloud infrastructure')
    })

    test('forbids sst dev', () => {
      expect(isIacForbidden('sst', ['dev'])).toContain('SST dev deploys infrastructure for dev mode')
    })

    test('forbids sst remove', () => {
      expect(isIacForbidden('sst', ['remove'])).toContain('Remove tears down SST-managed infrastructure')
    })

    test('forbids sst state rm', () => {
      expect(isIacForbidden('sst', ['state', 'rm', 'app.table'])).toContain('State mutation can orphan resources')
    })

    test('forbids sst secret set', () => {
      expect(isIacForbidden('sst', ['secret', 'set', 'API_KEY', 'value'])).toContain('Setting secrets mutates remote runtime configuration')
    })

    test('forbids sst secret remove', () => {
      expect(isIacForbidden('sst', ['secret', 'remove', 'API_KEY'])).toContain('Removing secrets can break runtime config')
    })

    test('allows sst state list', () => {
      expect(isIacForbidden('sst', ['state', 'list'])).toBeNull()
    })
  })

  describe('cdk', () => {
    test('allows cdk synth', () => {
      expect(isIacForbidden('cdk', ['synth'])).toBeNull()
    })

    test('allows cdk diff', () => {
      expect(isIacForbidden('cdk', ['diff'])).toBeNull()
    })

    test('allows cdk list', () => {
      expect(isIacForbidden('cdk', ['list'])).toBeNull()
    })

    test('allows cdk doctor', () => {
      expect(isIacForbidden('cdk', ['doctor'])).toBeNull()
    })

    test('allows cdk init --generate-only', () => {
      expect(isIacForbidden('cdk', ['init', '--generate-only'])).toBeNull()
    })

    test('forbids cdk deploy', () => {
      expect(isIacForbidden('cdk', ['deploy'])).toContain('CDK deploy mutates CloudFormation stacks')
    })

    test('forbids cdk destroy', () => {
      expect(isIacForbidden('cdk', ['destroy'])).toContain('CDK destroy tears down deployed infrastructure')
    })

    test('forbids cdk bootstrap', () => {
      expect(isIacForbidden('cdk', ['bootstrap'])).toContain('CDK bootstrap provisions account-level resources')
    })

    test('forbids cdk watch', () => {
      expect(isIacForbidden('cdk', ['watch'])).toContain('CDK watch can auto-deploy')
    })

    test('forbids cdk init without --generate-only', () => {
      expect(isIacForbidden('cdk', ['init', 'app', '--language', 'ts'])).toContain('CDK init without `--generate-only`')
    })
  })

  test('non-IaC tool returns null', () => {
    expect(isIacForbidden('npm', ['install'])).toBeNull()
  })
})