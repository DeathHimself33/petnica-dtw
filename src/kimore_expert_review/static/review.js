function setupReviewForm(form) {
  const labels = Array.from(form.querySelectorAll('input[name="execution_label"]'));
  const details = form.querySelector('[data-error-details]');
  const type = form.querySelector('select[name="error_type"]');
  const severity = form.querySelector('select[name="severity"]');
  const notes = form.querySelector('textarea[name="review_notes"]');
  const notesAlwaysRequired = notes ? notes.required : false;

  function sync() {
    const selected = labels.find((input) => input.checked)?.value;
    const isError = selected === 'error';
    const needsNote = selected === 'uncertain' || selected === 'ungradable';
    if (details) details.classList.toggle('is-disabled', !isError);
    for (const control of [type, severity]) {
      if (!control) continue;
      control.disabled = !isError;
      control.required = isError;
      if (!isError) control.value = '';
    }
    if (notes) notes.required = notesAlwaysRequired || needsNote;
  }

  for (const input of labels) input.addEventListener('change', sync);
  sync();
}

for (const form of document.querySelectorAll('[data-review-form]')) {
  setupReviewForm(form);
}

async function registerReviewTools() {
  const context = document.modelContext;
  if (!context?.registerTool) return;

  await context.registerTool({
    name: 'get_review_progress',
    title: 'Read expert-review progress',
    description: 'Read the active human reviewer progress without revealing labels or changing review state.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: false },
    async execute() {
      const response = await fetch('/api/progress', { headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error('A reviewer session is required.');
      return response.json();
    },
  });

  await context.registerTool({
    name: 'open_next_review_item',
    title: 'Open next review item',
    description: 'Navigate the active human reviewer to the next unfinished item. This never submits or changes a label.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: false },
    execute() {
      window.location.assign('/review');
      return { status: 'opening_next_item' };
    },
  });
}

registerReviewTools().catch(() => {
  // WebMCP is optional; the visible human workflow remains fully functional.
});
