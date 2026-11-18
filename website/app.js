const metricKeys = [
  "accuracy",
  "auc",
  "f1",
  "precision",
  "recall",
  "test_disc_dice",
  "test_disc_iou",
  "test_cup_dice",
  "test_cup_iou"
];

fetch("../results/website_results_partial.json")
  .then(response => {
    if (!response.ok) throw new Error("Could not load measured export");
    return response.json();
  })
  .then(payload => render(payload))
  .catch(error => {
    document.getElementById("experiments").innerHTML =
      `<article class="card"><h3>Export unavailable</h3><p>${error.message}</p></article>`;
  });

function render(payload) {
  const experiments = payload.experiments;
  const datasets = [...new Set(experiments.map(item => item.dataset))].sort();
  const filter = document.getElementById("dataset-filter");
  datasets.forEach(dataset => {
    const option = document.createElement("option");
    option.value = dataset;
    option.textContent = dataset;
    filter.appendChild(option);
  });

  document.getElementById("summary").innerHTML = [
    [String(experiments.length), "measured combinations"],
    [String(datasets.length), "datasets represented"],
    ["705", "ACRIMA images"],
    ["159", "RIM-ONE images"]
  ].map(([value, label]) =>
    `<article class="summary-card"><strong>${value}</strong><span>${label}</span></article>`
  ).join("");

  function draw() {
    const selected = filter.value;
    const visible = selected === "all"
      ? experiments
      : experiments.filter(item => item.dataset === selected);
    document.getElementById("experiments").innerHTML = visible.map(card).join("");
  }
  filter.addEventListener("change", draw);
  draw();
}

function card(experiment) {
  const metrics = experiment.metrics || {};
  const values = [];
  for (const key of metricKeys) {
    if (typeof metrics[key] === "number") {
      values.push([key, metrics[key]]);
    }
  }
  if (metrics.test_disc_dice != null) values.push(["disc Dice", metrics.test_disc_dice]);
  if (metrics.test_cup_dice != null) values.push(["cup Dice", metrics.test_cup_dice]);
  const metricHtml = values.slice(0, 4).map(([key, value]) =>
    `<div class="metric"><strong>${Number(value).toFixed(4)}</strong><span>${key}</span></div>`
  ).join("");
  return `<article class="card">
    <h3>${experiment.dataset}</h3>
    <span class="tag">${experiment.task}</span>
    <span class="tag">${experiment.split}</span>
    <div class="metrics">${metricHtml}</div>
  </article>`;
}
