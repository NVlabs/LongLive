// Fill these URLs when the paper and code are ready.
const PUBLICATION_URLS = {
  paper: "",
  code: "",
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
