(() => {
  "use strict";

  const figure = document.querySelector("#method-figure");
  if (!figure) {
    return;
  }

  const svgNamespace = "http://www.w3.org/2000/svg";
  const nodes = ["1", "2", "3", "4"];
  const edges = [
    { id: "12", a: "1", b: "2" },
    { id: "13", a: "1", b: "3" },
    { id: "14", a: "1", b: "4" },
    { id: "23", a: "2", b: "3" },
    { id: "24", a: "2", b: "4" },
    { id: "34", a: "3", b: "4" },
  ];

  const edgeById = new Map(edges.map((edge) => [edge.id, edge]));
  const softProbabilityLabel = "Soft edge probabilities \u03b2(x, \u03c4)";
  const hardMatrixLabel = "Training surrogate \u03b2(x, \u03c4)";
  const nodePositions = {
    1: [180, 34],
    2: [318, 116],
    3: [225, 246],
    4: [43, 173],
  };

  const totalEpochs = 10;
  const replayDuration = 5000;
  const trainingDuration = replayDuration;
  const trainingTemperatures = [1.10, 1.00, 0.90, 0.80, 0.68, 0.56, 0.44, 0.33, 0.23, 0.14];
  const trainingStates = {
    a: [
      { scores: { 12: 0.55, 13: 0.52, 14: 0.47, 23: 0.57, 24: 0.45, 34: 0.50 } },
      { scores: { 12: 0.63, 13: 0.56, 14: 0.50, 23: 0.61, 24: 0.43, 34: 0.48 } },
      { scores: { 12: 0.74, 13: 0.68, 14: 0.55, 23: 0.72, 24: 0.39, 34: 0.45 } },
      { scores: { 12: 0.78, 13: 0.71, 14: 0.60, 23: 0.76, 24: 0.36, 34: 0.42 } },
      { scores: { 12: 0.76, 13: 0.75, 14: 0.67, 23: 0.80, 24: 0.34, 34: 0.40 } },
      { scores: { 12: 0.82, 13: 0.80, 14: 0.75, 23: 0.68, 24: 0.31, 34: 0.36 } },
      { scores: { 12: 0.84, 13: 0.82, 14: 0.78, 23: 0.58, 24: 0.33, 34: 0.39 } },
      { scores: { 12: 0.88, 13: 0.85, 14: 0.81, 23: 0.48, 24: 0.29, 34: 0.34 } },
      { scores: { 12: 0.92, 13: 0.89, 14: 0.85, 23: 0.39, 24: 0.25, 34: 0.30 } },
      { scores: { 12: 0.94, 13: 0.91, 14: 0.88, 23: 0.31, 24: 0.22, 34: 0.27 } },
    ],
    b: [
      { scores: { 12: 0.54, 13: 0.50, 14: 0.47, 23: 0.56, 24: 0.51, 34: 0.53 } },
      { scores: { 12: 0.61, 13: 0.55, 14: 0.48, 23: 0.60, 24: 0.57, 34: 0.62 } },
      { scores: { 12: 0.70, 13: 0.58, 14: 0.51, 23: 0.67, 24: 0.69, 34: 0.57 } },
      { scores: { 12: 0.73, 13: 0.56, 14: 0.68, 23: 0.70, 24: 0.74, 34: 0.62 } },
      { scores: { 12: 0.76, 13: 0.50, 14: 0.61, 23: 0.79, 24: 0.53, 34: 0.75 } },
      { scores: { 12: 0.81, 13: 0.43, 14: 0.42, 23: 0.84, 24: 0.42, 34: 0.81 } },
      { scores: { 12: 0.83, 13: 0.40, 14: 0.34, 23: 0.86, 24: 0.40, 34: 0.82 } },
      { scores: { 12: 0.86, 13: 0.35, 14: 0.26, 23: 0.89, 24: 0.37, 34: 0.85 } },
      { scores: { 12: 0.88, 13: 0.32, 14: 0.20, 23: 0.91, 24: 0.35, 34: 0.87 } },
      { scores: { 12: 0.89, 13: 0.30, 14: 0.17, 23: 0.93, 24: 0.34, 34: 0.88 } },
    ],
  };

  const samples = [
    {
      id: "a",
      name: "Input A",
    },
    {
      id: "b",
      name: "Input B",
    },
  ];

  const slider = figure.querySelector("#tree-temperature");
  const temperatureValue = figure.querySelector("#temperature-value");
  const epochCounter = figure.querySelector("#epoch-counter");
  const sampleEpochCounters = figure.querySelectorAll(".sample-epoch");
  const epochProgress = figure.querySelector("#epoch-progress");
  const replayButton = figure.querySelector("#method-replay");
  const tooltip = figure.querySelector("#edge-tooltip");
  const status = figure.querySelector("#method-status");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const narrowViewport = window.matchMedia("(max-width: 700px)");
  const coarsePointer = window.matchMedia("(pointer: coarse)");
  const allTrees = enumerateSpanningTrees();
  const softMarginals = new Map();
  const currentScores = new Map();
  const hardTrees = new Map();

  let currentRepresentation = "scores";
  let highlightedEdge = null;
  let pinnedEdge = null;
  let replayToken = 0;
  let viewsRendered = false;

  initializeEpochProgress();
  setTrainingState(0, 0, Number(slider.value));
  samples.forEach((sample) => {
    renderGraph(`dependency-graph-${sample.id}`, sample);
    renderMatrix(`score-matrix-${sample.id}`, sample, "scores");
    renderMatrix(`dependency-matrix-${sample.id}`, sample, "soft");
  });

  viewsRendered = true;
  updateTemperature(Number(slider.value));
  setRepresentation("scores");

  slider.addEventListener("input", () => {
    replayToken += 1;
    showManualSoftState();
    updateTemperature(Number(slider.value));
    status.textContent = `Soft tree probabilities updated for temperature ${Number(slider.value).toFixed(2)}`;
  });
  replayButton.addEventListener("click", replay);
  figure.addEventListener("click", (event) => {
    if (!coarsePointer.matches || !pinnedEdge || event.target.closest("[data-edge], button, input, label, output, a, [role='button']")) {
      return;
    }
    pinnedEdge = null;
    clearHighlight();
  });
  window.addEventListener("resize", schedulePinnedTooltipReposition);
  window.addEventListener("orientationchange", schedulePinnedTooltipReposition);
  window.visualViewport?.addEventListener("resize", schedulePinnedTooltipReposition);
  showReadyState();

  function enumerateSpanningTrees() {
    const trees = [];
    for (let first = 0; first < edges.length - 2; first += 1) {
      for (let second = first + 1; second < edges.length - 1; second += 1) {
        for (let third = second + 1; third < edges.length; third += 1) {
          const candidate = [edges[first], edges[second], edges[third]];
          if (isTree(candidate)) {
            trees.push(candidate.map((edge) => edge.id));
          }
        }
      }
    }
    return trees;
  }

  function isTree(candidateEdges) {
    const parent = Object.fromEntries(nodes.map((node) => [node, node]));
    const find = (node) => {
      let root = node;
      while (parent[root] !== root) {
        root = parent[root];
      }
      while (parent[node] !== node) {
        const next = parent[node];
        parent[node] = root;
        node = next;
      }
      return root;
    };

    for (const edge of candidateEdges) {
      const firstRoot = find(edge.a);
      const secondRoot = find(edge.b);
      if (firstRoot === secondRoot) {
        return false;
      }
      parent[firstRoot] = secondRoot;
    }
    return new Set(nodes.map(find)).size === 1;
  }

  function maximumSpanningTree(scores) {
    const parent = Object.fromEntries(nodes.map((node) => [node, node]));
    const find = (node) => {
      if (parent[node] !== node) {
        parent[node] = find(parent[node]);
      }
      return parent[node];
    };
    const selected = [];
    const rankedEdges = [...edges].sort((first, second) => scores[second.id] - scores[first.id]);

    for (const edge of rankedEdges) {
      const firstRoot = find(edge.a);
      const secondRoot = find(edge.b);
      if (firstRoot === secondRoot) {
        continue;
      }
      selected.push(edge.id);
      parent[firstRoot] = secondRoot;
      if (selected.length === nodes.length - 1) {
        break;
      }
    }
    return new Set(selected);
  }

  function calculateSoftMarginals(scores, temperature) {
    const logWeights = allTrees.map((tree) => tree.reduce((total, edgeId) => total + scores[edgeId] / temperature, 0));
    const maximumLogWeight = Math.max(...logWeights);
    const weights = logWeights.map((logWeight) => Math.exp(logWeight - maximumLogWeight));
    const normalization = weights.reduce((total, weight) => total + weight, 0);
    const marginals = Object.fromEntries(edges.map((edge) => [edge.id, 0]));

    allTrees.forEach((tree, index) => {
      const probability = weights[index] / normalization;
      tree.forEach((edgeId) => {
        marginals[edgeId] += probability;
      });
    });
    return marginals;
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([attribute, value]) => element.setAttribute(attribute, String(value)));
    return element;
  }

  function svgText(attributes, text) {
    const element = svgElement("text", attributes);
    element.textContent = text;
    return element;
  }

  function renderGraph(targetId, sample) {
    const target = figure.querySelector(`#${targetId}`);
    const svg = svgElement("svg", {
      class: "latent-graph",
      viewBox: "20 10 320 260",
      role: "img",
      "aria-label": `${sample.name} illustrative latent dependence graph`,
    });
    const hardTree = hardTrees.get(sample.id);

    edges.forEach((edge, edgeIndex) => {
      const [startX, startY] = nodePositions[edge.a];
      const [endX, endY] = nodePositions[edge.b];
      const group = svgElement("g", {
        class: "graph-edge",
        "data-edge": edge.id,
        "data-sample": sample.id,
        tabindex: "0",
        role: "button",
        "aria-label": edgeAriaLabel(sample, edge.id),
        style: `--edge-index:${edgeIndex}`,
      });
      group.classList.toggle("is-hard-selected", hardTree.has(edge.id));

      const hitLine = svgElement("line", {
        class: "edge-hit",
        x1: startX,
        y1: startY,
        x2: endX,
        y2: endY,
      });
      const visualLine = svgElement("line", {
        class: "edge-line",
        x1: startX,
        y1: startY,
        x2: endX,
        y2: endY,
      });
      const hardAccent = svgElement("line", {
        class: "edge-hard-accent",
        x1: startX,
        y1: startY,
        x2: endX,
        y2: endY,
      });
      group.append(hitLine, hardAccent, visualLine);
      attachEdgeInteractions(group, sample.id, edge.id);
      svg.append(group);
    });

    nodes.forEach((node) => {
      const [x, y] = nodePositions[node];
      const nodeGroup = svgElement("g", {
        class: "graph-node",
        "data-node": node,
        "data-sample": sample.id,
      });
      nodeGroup.append(
        svgElement("circle", { class: "node-disc", cx: x, cy: y, r: 25 }),
        svgText({ class: "node-symbol", x, y: y + 6, "text-anchor": "middle" }, `z${node}`),
      );
      svg.append(nodeGroup);
    });

    target.replaceChildren(svg);
  }

  function renderMatrix(targetId, sample, representation) {
    const target = figure.querySelector(`#${targetId}`);
    const grid = document.createElement("div");
    grid.className = `score-matrix score-matrix--${representation}`;
    grid.setAttribute("role", "grid");
    grid.setAttribute("aria-label", `${sample.name} ${representation === "scores" ? "pairwise score" : "soft edge-marginal"} matrix`);
    grid.append(matrixLabel("", "matrix-corner"));
    nodes.forEach((node) => grid.append(matrixLabel(`z${node}`, "matrix-axis", "columnheader")));

    nodes.forEach((rowNode) => {
      grid.append(matrixLabel(`z${rowNode}`, "matrix-axis", "rowheader"));
      nodes.forEach((columnNode) => {
        if (rowNode === columnNode) {
          grid.append(matrixLabel("", "matrix-diagonal", "gridcell"));
          return;
        }

        const edgeId = edgeIdFor(rowNode, columnNode);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "matrix-cell";
        button.dataset.edge = edgeId;
        button.dataset.sample = sample.id;
        button.dataset.representation = representation;
        button.style.setProperty("--edge-index", String(edges.findIndex((edge) => edge.id === edgeId)));
        button.setAttribute("role", "gridcell");
        button.setAttribute("aria-label", edgeAriaLabel(sample, edgeId));
        button.append(Object.assign(document.createElement("span"), { className: "sr-only", textContent: `z${rowNode} to z${columnNode}` }));
        attachEdgeInteractions(button, sample.id, edgeId);
        grid.append(button);
      });
    });
    target.replaceChildren(grid);
  }

  function matrixLabel(text, className, role) {
    const label = document.createElement("span");
    label.className = className;
    if (role) {
      label.setAttribute("role", role);
    }
    label.textContent = text;
    return label;
  }

  function edgeIdFor(firstNode, secondNode) {
    return [firstNode, secondNode].sort().join("");
  }

  function updateTemperature(temperature) {
    temperatureValue.value = temperature.toFixed(2);
    temperatureValue.textContent = temperature.toFixed(2);
    slider.setAttribute("aria-valuetext", `${temperature.toFixed(2)} temperature`);
    samples.forEach((sample) => softMarginals.set(sample.id, calculateSoftMarginals(currentScores.get(sample.id), temperature)));

    if (currentRepresentation !== "scores") {
      applyRepresentation();
    }
    if (highlightedEdge) {
      showTooltip(highlightedEdge.sampleId, highlightedEdge.edgeId, highlightedEdge.anchor);
    }
  }

  function setRepresentation(representation) {
    currentRepresentation = representation;
    applyRepresentation();
  }

  function applyRepresentation() {
    samples.forEach((sample) => {
      updateGraph(sample);
      updateScoreMatrix(sample);
      updateSoftMatrix(sample);
    });
  }

  function updateGraph(sample) {
    const hardTree = hardTrees.get(sample.id);
    figure.querySelectorAll(`#dependency-graph-${sample.id} .graph-edge`).forEach((group) => {
      const edgeId = group.dataset.edge;
      const line = group.querySelector(".edge-line");
      const isHardEdge = hardTree.has(edgeId);
      group.classList.toggle("is-hard-selected", isHardEdge);
      if (isHardEdge) {
        line.style.stroke = "#0d7d77";
        line.style.strokeWidth = "5px";
        line.style.opacity = "1";
      } else {
        line.style.stroke = "transparent";
        line.style.strokeWidth = "1px";
        line.style.opacity = "0";
      }

      line.style.strokeDasharray = "none";
      group.setAttribute("aria-label", edgeAriaLabel(sample, edgeId));
    });
  }

  function updateScoreMatrix(sample) {
    const values = currentScores.get(sample.id);
    const target = figure.querySelector(`#score-matrix-${sample.id}`);
    target.querySelectorAll(".matrix-cell").forEach((cell) => {
      const value = values[cell.dataset.edge];
      cell.style.setProperty("--strength", value.toFixed(4));
      cell.setAttribute("aria-label", `${edgeAriaLabel(sample, cell.dataset.edge)}. Pairwise encoder score ${value.toFixed(2)}.`);
    });
  }

  function updateSoftMatrix(sample) {
    const values = softMarginals.get(sample.id);
    const target = figure.querySelector(`#dependency-matrix-${sample.id}`);
    target.querySelectorAll(".matrix-cell").forEach((cell) => {
      const value = values[cell.dataset.edge];
      cell.style.setProperty("--strength", value.toFixed(4));
      cell.setAttribute("aria-label", `${edgeAriaLabel(sample, cell.dataset.edge)}. ${softProbabilityLabel}.`);
    });
    target.querySelector(".score-matrix").setAttribute("aria-label", `${sample.name} soft edge-marginal matrix`);
  }

  function edgeAriaLabel(sample, edgeId) {
    const edge = edgeById.get(edgeId);
    const hardState = hardTrees.get(sample.id).has(edgeId) ? "selected by the hard MWST" : "not selected by the hard MWST";
    const softProbability = softMarginals.get(sample.id)?.[edgeId] ?? 0;
    return `Illustrative method example. z${edge.a} to z${edge.b}; pairwise score ${currentScores.get(sample.id)[edgeId].toFixed(2)}; soft inclusion probability ${softProbability.toFixed(2)}; ${hardState}`;
  }

  function attachEdgeInteractions(element, sampleId, edgeId) {
    element.addEventListener("pointerenter", () => activateEdge(sampleId, edgeId, element));
    element.addEventListener("pointerleave", () => {
      if (!pinnedEdge) {
        clearHighlight();
      }
    });
    element.addEventListener("focus", () => activateEdge(sampleId, edgeId, element));
    element.addEventListener("blur", () => {
      if (!pinnedEdge) {
        clearHighlight();
      }
    });
    element.addEventListener("click", (event) => {
      event.preventDefault();
      if (pinnedEdge && pinnedEdge.sampleId === sampleId && pinnedEdge.edgeId === edgeId) {
        pinnedEdge = null;
        clearHighlight();
        return;
      }
      pinnedEdge = { sampleId, edgeId, anchor: element };
      activateEdge(sampleId, edgeId, element);
    });
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        element.click();
      }
    });
  }

  function activateEdge(sampleId, edgeId, anchor) {
    highlightedEdge = { sampleId, edgeId, anchor };
    updateHighlights();
    showTooltip(sampleId, edgeId, anchor);
  }

  function clearHighlight() {
    highlightedEdge = null;
    updateHighlights();
    tooltip.hidden = true;
  }

  function updateHighlights() {
    figure.querySelectorAll("[data-edge]").forEach((element) => {
      const isHighlighted = highlightedEdge
        && element.dataset.sample === highlightedEdge.sampleId
        && element.dataset.edge === highlightedEdge.edgeId;
      element.classList.toggle("is-highlighted", Boolean(isHighlighted));
      element.classList.toggle("is-pinned", Boolean(isHighlighted && pinnedEdge));
    });

    figure.querySelectorAll(".graph-node").forEach((element) => {
      const edge = highlightedEdge ? edgeById.get(highlightedEdge.edgeId) : null;
      const isConnected = edge
        && element.dataset.sample === highlightedEdge.sampleId
        && (element.dataset.node === edge.a || element.dataset.node === edge.b);
      element.classList.toggle("is-connected", Boolean(isConnected));
    });
  }

  function showTooltip(sampleId, edgeId, anchor) {
    const sample = samples.find((candidate) => candidate.id === sampleId);
    const edge = edgeById.get(edgeId);
    const softProbability = softMarginals.get(sampleId)[edgeId];
    const hardState = hardTrees.get(sampleId).has(edgeId) ? "selected" : "not selected";
    tooltip.textContent = `Illustrative method example\nz${edge.a} - z${edge.b}\n\npairwise score: ${currentScores.get(sampleId)[edgeId].toFixed(2)}\nsoft P(edge): ${softProbability.toFixed(2)}\nhard MWST: ${hardState}`;
    tooltip.hidden = false;

    const rect = anchor.getBoundingClientRect();
    if (narrowViewport.matches || coarsePointer.matches) {
      positionMobileTooltip(rect);
      return;
    }
    const tooltipWidth = 210;
    const left = Math.max(12, Math.min(rect.left + rect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - 12));
    const top = Math.max(12, rect.top - 135);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function positionMobileTooltip(anchorRect) {
    const viewportMargin = 12;
    const tooltipRect = tooltip.getBoundingClientRect();
    const maximumLeft = Math.max(viewportMargin, window.innerWidth - tooltipRect.width - viewportMargin);
    const maximumTop = Math.max(viewportMargin, window.innerHeight - tooltipRect.height - viewportMargin);
    const centeredLeft = anchorRect.left + anchorRect.width / 2 - tooltipRect.width / 2;
    const aboveTop = anchorRect.top - tooltipRect.height - 10;
    const preferredTop = aboveTop >= viewportMargin ? aboveTop : anchorRect.bottom + 10;
    const left = Math.max(viewportMargin, Math.min(centeredLeft, maximumLeft));
    const top = Math.max(viewportMargin, Math.min(preferredTop, maximumTop));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function repositionPinnedTooltip() {
    if (!coarsePointer.matches || !pinnedEdge) {
      return;
    }
    if (!pinnedEdge.anchor.isConnected) {
      pinnedEdge = null;
      clearHighlight();
      return;
    }
    showTooltip(pinnedEdge.sampleId, pinnedEdge.edgeId, pinnedEdge.anchor);
  }

  function schedulePinnedTooltipReposition() {
    window.requestAnimationFrame(repositionPinnedTooltip);
  }

  function setSampling(isSampling) {
    figure.classList.toggle("is-sampling", isSampling);
  }

  function showManualSoftState() {
    figure.classList.remove("is-replaying", "is-complete", "is-hard-revealed", "feedback-pulse-active", "is-training");
    figure.classList.add("is-scores-revealed");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.add("is-visible"));
    setSampling(false);
    setRepresentation("soft");
  }

  function showReadyState() {
    const highTemperature = trainingTemperatures[0];
    figure.classList.add("is-replaying");
    figure.classList.remove("is-complete", "is-hard-revealed", "feedback-pulse-active", "is-training");
    figure.classList.add("is-scores-revealed");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.add("is-visible"));
    slider.value = highTemperature.toFixed(2);
    setTrainingState(0, 0, highTemperature);
    updateTemperature(highTemperature);
    updateEpochCounter(1);
    setRepresentation("scores");
    slider.disabled = true;
    replayButton.textContent = "Play animation";
    status.textContent = "Ready to animate input-specific dependence learning";
  }

  async function replay() {
    const token = ++replayToken;
    const highTemperature = trainingTemperatures[0];
    figure.classList.add("is-replaying");
    figure.classList.remove("is-complete", "is-hard-revealed", "feedback-pulse-active");
    figure.classList.add("is-scores-revealed");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.add("is-visible"));
    pinnedEdge = null;
    clearHighlight();
    setSampling(false);
    replayButton.disabled = true;
    slider.disabled = true;
    slider.value = highTemperature.toFixed(2);
    setTrainingState(0, 0, highTemperature);
    updateTemperature(highTemperature);
    updateEpochCounter(1);
    setRepresentation("scores");

    setRepresentation("soft");
    status.textContent = "Learning input-specific dependence structures over 10 epochs";
    if (!await animateEpochs(token, trainingDuration)) {
      figure.classList.remove("is-training");
      return;
    }
    figure.classList.remove("is-training");

    figure.classList.remove("is-replaying");
    figure.classList.add("is-complete");
    replayButton.disabled = false;
    slider.disabled = false;
    replayButton.textContent = "Replay";
    status.textContent = "Interactive method visualization ready";
  }

  async function revealStep(step, message, duration, token) {
    if (token !== replayToken) {
      return false;
    }
    figure.querySelectorAll(`[data-step="${step}"]`).forEach((element) => element.classList.add("is-visible"));
    status.textContent = message;
    return pause(duration, token);
  }

  function pause(duration, token) {
    if (reducedMotion.matches || duration === 0) {
      return Promise.resolve(token === replayToken);
    }
    return new Promise((resolve) => {
      window.setTimeout(() => resolve(token === replayToken), duration);
    });
  }

  function updateEpochCounter(epoch) {
    epochCounter.textContent = `Epoch ${epoch} / ${totalEpochs}`;
    sampleEpochCounters.forEach((counter) => {
      counter.textContent = `Epoch ${epoch}`;
    });
    epochProgress.querySelectorAll(".epoch-progress-dot").forEach((dot, index) => {
      dot.classList.toggle("is-complete", index < epoch - 1);
      dot.classList.toggle("is-active", index === epoch - 1);
    });
    const activeDot = epochProgress.querySelector(".epoch-progress-dot.is-active");
    if (activeDot) {
      activeDot.classList.remove("is-entering");
      void activeDot.offsetWidth;
      activeDot.classList.add("is-entering");
    }
  }

  function animateEpochs(token, duration) {
    return new Promise((resolve) => {
      if (reducedMotion.matches) {
        slider.value = trainingTemperatures[totalEpochs - 1].toFixed(2);
        setTrainingState(totalEpochs - 1, 0, trainingTemperatures[totalEpochs - 1]);
        updateTemperature(trainingTemperatures[totalEpochs - 1]);
        updateEpochCounter(totalEpochs);
        resolve(token === replayToken);
        return;
      }

      let epochIndex = 0;
      const epochDuration = duration / (totalEpochs - 1);
      const trainingStart = performance.now();
      const animateTransition = () => {
        if (token !== replayToken) {
          resolve(false);
          return;
        }
        const transitionStart = performance.now();
        const transitionDuration = Math.max(
          0,
          trainingStart + epochDuration * (epochIndex + 1) - transitionStart,
        );
        const renderFrame = (now) => {
          if (token !== replayToken) {
            resolve(false);
            return;
          }
          const progress = Math.min((now - transitionStart) / transitionDuration, 1);
          const easedProgress = 1 - (1 - progress) ** 3;
          const temperature = interpolateValue(
            trainingTemperatures[epochIndex],
            trainingTemperatures[epochIndex + 1],
            easedProgress,
          );
          slider.value = temperature.toFixed(2);
          setTrainingState(epochIndex, easedProgress, temperature);
          updateTemperature(temperature);

          if (progress < 1) {
            window.requestAnimationFrame(renderFrame);
            return;
          }

          epochIndex += 1;
          updateEpochCounter(epochIndex + 1);
          if (epochIndex < totalEpochs - 1) {
            animateTransition();
            return;
          }
          resolve(token === replayToken);
        };
        window.requestAnimationFrame(renderFrame);
      };
      figure.classList.add("is-training");
      animateTransition();
    });
  }

  function setTrainingState(epochIndex, interpolation, temperature) {
    samples.forEach((sample) => {
      const fromScores = trainingStates[sample.id][epochIndex].scores;
      const toScores = trainingStates[sample.id][Math.min(epochIndex + 1, totalEpochs - 1)].scores;
      const scores = Object.fromEntries(edges.map((edge) => [
        edge.id,
        fromScores[edge.id] + (toScores[edge.id] - fromScores[edge.id]) * interpolation,
      ]));
      currentScores.set(sample.id, scores);
      const hardTree = maximumSpanningTree(scores);
      hardTrees.set(sample.id, hardTree);
      softMarginals.set(sample.id, calculateSoftMarginals(scores, temperature));
      if (viewsRendered) {
        updateHardTreeMembership(sample, hardTree);
      }
    });
    if (viewsRendered && currentRepresentation) {
      applyRepresentation();
    }
  }

  function initializeEpochProgress() {
    for (let index = 0; index < totalEpochs; index += 1) {
      const dot = document.createElement("span");
      dot.className = "epoch-progress-dot";
      epochProgress.append(dot);
    }
  }

  function updateHardTreeMembership(sample, hardTree) {
    figure.querySelectorAll(`#dependency-graph-${sample.id} .graph-edge`).forEach((group) => {
      group.classList.toggle("is-hard-selected", hardTree.has(group.dataset.edge));
    });
  }

  function interpolateValue(start, end, progress) {
    return start + (end - start) * progress;
  }

  function sameTree(firstTree, secondTree) {
    return firstTree.size === secondTree.size && [...firstTree].every((edgeId) => secondTree.has(edgeId));
  }
})();