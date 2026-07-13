'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync } = require('child_process');

const BINARY_DIR = path.join(os.homedir(), '.magnitude', 'bin');
const REPO = 'magnitudedev/magnitude';

const ASSETS = {
  cli: {
    displayName: 'Magnitude',
    executableBase: 'magnitude',
    versionFile: 'magnitude.version',
    assetPrefix: 'magnitude',
  },
  acn: {
    displayName: 'Magnitude ACN',
    executableBase: 'magnitude-acn',
    versionFile: 'magnitude-acn.version',
    assetPrefix: 'magnitude-acn',
  },
};

function getPlatformKey() {
  const platform = process.platform;
  const arch = process.arch;

  const map = {
    'darwin-arm64': 'darwin-arm64',
    'darwin-x64': 'darwin-x64',
    'linux-x64': 'linux-x64',
    'linux-arm64': 'linux-arm64',
    'win32-x64': 'windows-x64',
  };

  const key = `${platform}-${arch}`;
  if (!map[key]) {
    throw new Error(`Unsupported platform: ${key}. Magnitude supports: ${Object.keys(map).join(', ')}`);
  }
  return map[key];
}

function executableName(kind) {
  const ext = process.platform === 'win32' ? '.exe' : '';
  return `${ASSETS[kind].executableBase}${ext}`;
}

function executablePath(kind) {
  return path.join(BINARY_DIR, executableName(kind));
}

function versionPath(kind) {
  return path.join(BINARY_DIR, ASSETS[kind].versionFile);
}

function versionMatches(kind, version) {
  try {
    const cached = fs.readFileSync(versionPath(kind), 'utf8').trim();
    return cached === version;
  } catch {
    return false;
  }
}

function releaseTag(version) {
  return `@magnitudedev/cli@${version}`;
}

function releaseBaseUrl() {
  return (process.env.MAGNITUDE_RELEASE_BASE_URL || `https://github.com/${REPO}/releases/download`).replace(/\/+$/, '');
}

function assetName(kind, platformKey = getPlatformKey()) {
  return `${ASSETS[kind].assetPrefix}-${platformKey}.tar.gz`;
}

function assetUrl(kind, version, platformKey = getPlatformKey()) {
  return `${releaseBaseUrl()}/${encodeURIComponent(releaseTag(version))}/${assetName(kind, platformKey)}`;
}

function download(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) {
      reject(new Error(`Too many redirects for ${url}`));
      return;
    }

    const parsed = new URL(url);
    const transport = parsed.protocol === 'http:' ? http : https;
    const req = transport.get(parsed, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301 || res.statusCode === 307 || res.statusCode === 308) {
        const location = res.headers.location;
        if (!location) {
          reject(new Error(`Download redirect missing Location header: ${url}`));
          return;
        }
        const nextUrl = new URL(location, url).toString();
        download(nextUrl, redirects + 1).then(resolve, reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`Download failed for ${url}: HTTP ${res.statusCode}`));
        return;
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
    req.on('error', reject);
  });
}

function smokeVersion(kind, binPath, version) {
  const args = kind === 'cli' ? ['--version'] : ['version'];
  const output = execFileSync(binPath, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  if (output !== version) {
    throw new Error(`${ASSETS[kind].displayName} version mismatch: expected ${version}, got ${output}`);
  }
}

async function downloadAndInstall(kind, version) {
  const platformKey = getPlatformKey();
  const fileName = assetName(kind, platformKey);
  const url = assetUrl(kind, version, platformKey);

  fs.mkdirSync(BINARY_DIR, { recursive: true });

  console.log(`Downloading ${ASSETS[kind].displayName} v${version} for ${platformKey}...`);

  const data = await download(url);
  const tmpFile = path.join(BINARY_DIR, `${fileName}.tmp`);
  fs.writeFileSync(tmpFile, data);

  try {
    const tarFlag = process.platform === 'win32' ? '-xf' : '-xzf';
    execFileSync('tar', [tarFlag, tmpFile, '-C', BINARY_DIR], { stdio: 'ignore' });

    const binPath = executablePath(kind);
    if (!fs.existsSync(binPath)) {
      throw new Error(`Binary not found after extraction at ${binPath}`);
    }

    if (process.platform !== 'win32') {
      fs.chmodSync(binPath, 0o755);
    }

    smokeVersion(kind, binPath, version);
    fs.writeFileSync(versionPath(kind), version, 'utf8');
    return binPath;
  } finally {
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

async function ensureAsset(kind, version) {
  const binPath = executablePath(kind);

  if (fs.existsSync(binPath) && versionMatches(kind, version)) {
    return binPath;
  }

  return downloadAndInstall(kind, version);
}

async function ensureBinaries(version, options = {}) {
  const binaryPath = await ensureAsset('cli', version);

  let acnPath = null;
  let acnError = null;
  if (options.prefetchAcn !== false) {
    try {
      acnPath = await ensureAsset('acn', version);
    } catch (error) {
      acnError = error;
      if (options.requireAcn) throw error;
    }
  }

  return { binaryPath, acnPath, acnError };
}

async function ensureBinary(version) {
  const result = await ensureBinaries(version);
  return result.binaryPath;
}

module.exports = {
  ensureAsset,
  ensureBinary,
  ensureBinaries,
  assetName,
  assetUrl,
  releaseTag,
};
