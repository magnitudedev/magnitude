'use strict';

const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const BINARY_DIR = path.join(os.homedir(), '.magnitude', 'bin');
const VERSION_FILE = path.join(BINARY_DIR, 'magnitude.version');
const REPO = 'magnitudedev/magnitude';

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

function getBinaryPath() {
  const ext = process.platform === 'win32' ? '.exe' : '';
  return path.join(BINARY_DIR, `magnitude${ext}`);
}

function versionMatches(version) {
  try {
    const cached = fs.readFileSync(VERSION_FILE, 'utf8').trim();
    return cached === version;
  } catch {
    return false;
  }
}

function download(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        return download(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`Download failed: HTTP ${res.statusCode}`));
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function downloadAndInstall(version) {
  const platformKey = getPlatformKey();
  const fileName = `magnitude-${platformKey}.tar.gz`;
  const url = `https://github.com/${REPO}/releases/download/%40magnitudedev%2Fcli%40${version}/${fileName}`;
  
  fs.mkdirSync(BINARY_DIR, { recursive: true });
  
  console.log(`Downloading Magnitude v${version} for ${platformKey}...`);
  
  const data = await download(url);
  const tmpFile = path.join(BINARY_DIR, fileName);
  fs.writeFileSync(tmpFile, data);
  
  try {
    if (process.platform === 'win32') {
      execSync(`tar -xf "${tmpFile}" -C "${BINARY_DIR}"`, { stdio: 'ignore' });
    } else {
      execSync(`tar -xzf "${tmpFile}" -C "${BINARY_DIR}"`, { stdio: 'ignore' });
    }
    
    const binPath = getBinaryPath();
    if (!fs.existsSync(binPath)) {
      throw new Error(`Binary not found after extraction at ${binPath}`);
    }
    
    if (process.platform !== 'win32') {
      fs.chmodSync(binPath, 0o755);
    }
    
    fs.writeFileSync(VERSION_FILE, version, 'utf8');
    return binPath;
  } finally {
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

async function ensureBinary(version) {
  const binPath = getBinaryPath();
  
  if (fs.existsSync(binPath) && versionMatches(version)) {
    return binPath;
  }
  
  return downloadAndInstall(version);
}

module.exports = { ensureBinary };