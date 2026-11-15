// Список документов СудАкт по ст. 290. Запуск в консоли браузера на sudact.ru/regular/.
// Пример: await sampleListing([[2020, 'H1'], [2020, 'H2']], 2) → corpus/sudact/listing.json

const LAW = 'Статья 290. Получение взятки(УК РФ)';
const PAUSE_MS = 3000;
const PAGE_SIZE = 10;
const MAX_PAGES = 50; // лимит сайта

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function periodRange(year, part) {
  if (part === 'H1') return [`01.01.${year}`, `30.06.${year}`];
  if (part === 'H2') return [`01.07.${year}`, `31.12.${year}`];
  return [`01.01.${year}`, `31.12.${year}`];
}

async function listPage(from, to, page, stage = '10') {
  const url = `/regular/doc_ajax/?page=${page}&regular-txt=&regular-case_doc=`
    + `&regular-lawchunkinfo=${encodeURIComponent(LAW)}`
    + `&regular-date_from=${from}&regular-date_to=${to}&regular-workflow_stage=${stage}`
    + '&regular-area=&regular-court=&regular-judge=';
  for (let attempt = 0; attempt < 15; attempt++) {
    const res = await fetch(`${url}&_=${Date.now()}`, { headers: { 'X-Requested-With': 'XMLHttpRequest' } });
    const json = await res.json();
    if (json.status === 'finished' || json.search_status === 'finished') {
      const doc = new DOMParser().parseFromString(json.content || '', 'text/html');
      const items = [...doc.querySelectorAll('li')].map((li) => {
        const a = li.querySelector('h4 a');
        const m = a && a.getAttribute('href').match(/\/regular\/doc\/([A-Za-z0-9]+)\//);
        if (!m) return null;
        return {
          id: m[1],
          title: a.textContent.trim(),
          court: (li.querySelector('.b-justice')?.textContent || '').trim().replace(/\s+/g, ' '),
        };
      }).filter(Boolean);
      return { total: Number((json.total_found || '').replace(/\D/g, '')), items };
    }
    await sleep(1500);
  }
  return { total: null, items: [] };
}

// periods: [[год, 'H1' | 'H2' | 'Y']]; pagesPerPeriod — случайных страниц на период.
async function sampleListing(periods, pagesPerPeriod) {
  const out = [];
  for (const [year, part] of periods) {
    const [from, to] = periodRange(year, part);
    const first = await listPage(from, to, 1);
    await sleep(PAUSE_MS);
    const pageCount = Math.min(MAX_PAGES, Math.ceil((first.total || 0) / PAGE_SIZE));
    const pages = new Set();
    while (pages.size < Math.min(pagesPerPeriod, pageCount)) {
      pages.add(1 + Math.floor(Math.random() * pageCount));
    }
    for (const page of pages) {
      const r = page === 1 ? first : await listPage(from, to, page);
      if (page !== 1) await sleep(PAUSE_MS);
      for (const item of r.items) out.push({ ...item, period: `${year}-${part}`, page, total: first.total });
    }
    if (first.items.length === 0) out.push({ id: null, period: `${year}-${part}`, page: 1, total: first.total, error: 'empty listing' });
  }
  return out;
}
