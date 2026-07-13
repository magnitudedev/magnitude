import { describe, test, expect } from 'bun:test'
import {
  isSysadminForbidden,
  getSysadminAlwaysForbiddenReason,
  isPackageManagerForbidden,
  SYSADMIN_ALWAYS_FORBIDDEN,
} from '../tools/sysadmin'

function split(command: string): { base: string, args: string[] } {
  const [base = '', ...args] = command.split(/\s+/).filter(Boolean)
  return { base, args }
}

describe('sysadmin tool policy', () => {
  describe('always-forbidden sysadmin commands', () => {
    test('power commands are in always-forbidden set with power reason', () => {
      expect(SYSADMIN_ALWAYS_FORBIDDEN.has('shutdown')).toBe(true)
      expect(SYSADMIN_ALWAYS_FORBIDDEN.has('reboot')).toBe(true)
      expect(getSysadminAlwaysForbiddenReason('shutdown')).toContain('power-control')
    })

    test('partition commands return partition reason', () => {
      expect(SYSADMIN_ALWAYS_FORBIDDEN.has('fdisk')).toBe(true)
      expect(getSysadminAlwaysForbiddenReason('fdisk')).toContain('Partition edits')
    })

    test('firewall commands return firewall reason', () => {
      expect(SYSADMIN_ALWAYS_FORBIDDEN.has('iptables')).toBe(true)
      expect(getSysadminAlwaysForbiddenReason('iptables')).toContain('Firewall mutations')
    })
  })

  describe('systemctl/service', () => {
    for (const cmd of [
      'systemctl rescue',
      'systemctl emergency',
      'systemctl stop sshd',
      'systemctl disable NetworkManager',
      'systemctl mask docker',
      'service sshd stop',
    ]) {
      test(`${cmd} is forbidden`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).not.toBeNull()
      })
    }

    for (const cmd of [
      'systemctl status sshd',
      'systemctl restart my-local-dev-service',
      'service myapp status',
    ]) {
      test(`${cmd} is allowed`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).toBeNull()
      })
    }
  })

  describe('kill/pkill/killall', () => {
    for (const cmd of [
      'kill -9 1',
      'kill -1',
      'kill -- -1',
      'pkill -9 node',
      'killall -9 python',
    ]) {
      test(`${cmd} is forbidden`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).not.toBeNull()
      })
    }

    for (const cmd of [
      'kill -9 1234',
      'pkill node',
      'killall -15 node',
    ]) {
      test(`${cmd} is allowed`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).toBeNull()
      })
    }
  })

  describe('mount/umount', () => {
    for (const cmd of [
      'mount /dev/disk1s1 /usr',
      'umount /var',
      'mount -o rw,remount /',
    ]) {
      test(`${cmd} is forbidden`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).not.toBeNull()
      })
    }

    for (const cmd of [
      'mount /dev/disk2 /Volumes/External',
      'umount /Volumes/External',
    ]) {
      test(`${cmd} is allowed`, () => {
        const { base, args } = split(cmd)
        expect(isSysadminForbidden(base, args)).toBeNull()
      })
    }
  })

  describe('package managers', () => {
    for (const cmd of [
      'apt-get remove nginx',
      'apt purge openssh-server',
      'yum autoremove',
      'dnf full-upgrade',
      'snap remove core',
      'brew uninstall jq',
      'brew cleanup',
      'brew services stop postgres',
      'brew services cleanup',
    ]) {
      test(`${cmd} is forbidden`, () => {
        const { base, args } = split(cmd)
        expect(isPackageManagerForbidden(base, args)).not.toBeNull()
      })
    }

    for (const cmd of [
      'brew install jq',
      'apt-get install jq',
      'apt search jq',
      'brew services list',
    ]) {
      test(`${cmd} is allowed`, () => {
        const { base, args } = split(cmd)
        expect(isPackageManagerForbidden(base, args)).toBeNull()
      })
    }
  })
})