// Publication links; leave unreleased resources empty.
const PUBLICATION_URLS = {
  paper: "https://arxiv.org/abs/2609.38154",
  code: "https://github.com/NVlabs/LongLive/tree/main/LongLive-Plug",
};
for (const [kind, url] of Object.entries(PUBLICATION_URLS)) {
  const link = document.getElementById(`${kind}-link`);
  if (!url.trim()) continue;
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.removeAttribute("aria-disabled");
  link.removeAttribute("title");
}
