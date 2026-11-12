// photo-geolocator-01 entry: a vision-model geolocator with a deterministic fallback.
// For each of the instance's 25 photos it asks claude-sonnet-5-5, once, where the photo
// was taken, with a structured reasoning prompt (visible clues -> country -> region ->
// coordinates), reads the latitude and longitude out of the answer, and posts every
// answer it read to the env at the end. A photo whose model call the gateway refuses
// (no provider key pushed, budget, request cap: a refusal is final) or whose answer
// holds no coordinates is NOT left unanswered: it gets a deterministic prior guess
// derived from the photo's own bytes, so every photo is always answered.
//
// It runs as the grader runs an entry, `node /bot/main.ts` on Node 22 (erasable
// TypeScript, the standard library only), with the trial's variables:
// MAKEFUN_ENV_URL (the env), ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY (the gateway
// and the dummy key it swaps for the real one).
//
// Whose key pays: every call runs on the Anthropic key pushed to the competition
// before submission (ANTHROPIC_API_KEY in a .env file, `makefun secrets push
// photo-geolocator-01 --env-file .env`): the house pays for none of an entry's
// model calls. With no Anthropic key pushed every call is answered 401 and every
// photo falls back to its prior guess.
//
// Deterministic given the model's answers: every request is a function of the
// photo's bytes and the fixed prompt, the calls run four at a time and their
// answers are posted in photo order, so a replay sends the same calls and leaves
// the same answers. The fallback is a pure function of the photo's bytes too.

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 1024;
const CONCURRENCY = 4;
const PROMPT = [
  'You are a geolocation expert. Study this street-level photo carefully.',
  'Reason briefly from visible clues: language and script on signs, driving side,',
  'license plates, road markings, architecture, vegetation, terrain, clothing,',
  'and any landmarks. Narrow it down: country, then region, then city or area.',
  'Then give your best estimate as one line of JSON: {"lat": <degrees north>, "lon": <degrees east>}.',
  'Put 2-4 short reasoning lines first, then the JSON line last.',
].join(' ');
/** Retried with a fixed backoff: a provider's transient errors. A gateway refusal (4xx) is final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

/** Population centers used as a deterministic prior when a photo has no model answer. */
const PRIORS: { lat: number; lon: number }[] = [
  { lat: 35.68, lon: 139.69 }, // Tokyo
  { lat: 28.61, lon: 77.21 }, // Delhi
  { lat: 31.23, lon: 121.47 }, // Shanghai
  { lat: -23.55, lon: -46.63 }, // Sao Paulo
  { lat: 19.43, lon: -99.13 }, // Mexico City
  { lat: 30.04, lon: 31.24 }, // Cairo
  { lat: 19.08, lon: 72.88 }, // Mumbai
  { lat: 39.9, lon: 116.41 }, // Beijing
  { lat: 23.81, lon: 90.41 }, // Dhaka
  { lat: 34.69, lon: 135.5 }, // Osaka
  { lat: 40.71, lon: -74.01 }, // New York
  { lat: 24.86, lon: 67.01 }, // Karachi
  { lat: -34.6, lon: -58.38 }, // Buenos Aires
  { lat: 41.01, lon: 28.98 }, // Istanbul
  { lat: 6.52, lon: 3.37 }, // Lagos
  { lat: 51.51, lon: -0.13 }, // London
  { lat: 48.85, lon: 2.35 }, // Paris
  { lat: 55.76, lon: 37.62 }, // Moscow
  { lat: -6.21, lon: 106.85 }, // Jakarta
  { lat: 13.76, lon: 100.5 }, // Bangkok
  { lat: 34.05, lon: -118.24 }, // Los Angeles
  { lat: 40.42, lon: -3.7 }, // Madrid
  { lat: -33.87, lon: 151.21 }, // Sydney
  { lat: 43.65, lon: -79.38 }, // Toronto
  { lat: 52.52, lon: 13.4 }, // Berlin
];

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** FNV-1a hash of bytes, for the deterministic prior. */
function fnv1a(bytes: Buffer): number {
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** The prior guess for a photo: a pure function of its bytes. */
function priorGuess(jpeg: Buffer): { lat: number; lon: number } {
  return PRIORS[fnv1a(jpeg) % PRIORS.length]!;
}

/** The model's answer to one photo, as text; null when the gateway refused the call (final). */
async function ask(jpeg: Buffer): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`agent: the model call was refused (HTTP ${res.status}): ${text.slice(0, 300)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let refused = 0;
  let noCoords = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      const text = await ask(jpeg);
      if (text === null) {
        refused += 1;
        const prior = priorGuess(jpeg);
        guesses[i] = { id: p.id, lat: prior.lat, lon: prior.lon };
        continue;
      }
      const c = coordinates(text);
      if (c === null) {
        noCoords += 1;
        process.stderr.write(`agent: no coordinates in the answer for ${p.id}: ${text.slice(0, 200)}\n`);
        const prior = priorGuess(jpeg);
        guesses[i] = { id: p.id, lat: prior.lat, lon: prior.lon };
      } else guesses[i] = { id: p.id, lat: c.lat, lon: c.lon };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return refused > 0 ? 1 : 0;
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  process.stdout.write(
    `agent: ${answers.length} of ${photos.length} photos answered (${refused} model refusals, ${noCoords} without coordinates, all fallen back to prior): HTTP ${res.status} ${await res.text()}`,
  );
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`agent: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
