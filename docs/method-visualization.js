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

  // Deliberately schematic scores make the contrast between the two topologies clear.
  const samples = [
    {
      id: "a",
      name: "Input A",
      demoScores: { 12: 0.94, 13: 0.88, 14: 0.84, 23: 0.31, 24: 0.22, 34: 0.27 },
    },
    {
      id: "b",
      name: "Input B",
      demoScores: { 12: 0.88, 13: 0.3, 14: 0.17, 23: 0.92, 24: 0.35, 34: 0.86 },
    },
  ];

  const slider = figure.querySelector("#tree-temperature");
  const temperatureValue = figure.querySelector("#temperature-value");
  const epochCounter = figure.querySelector("#epoch-counter");
  const replayButton = figure.querySelector("#method-replay");
  const tooltip = figure.querySelector("#edge-tooltip");
  const status = figure.querySelector("#method-status");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const allTrees = enumerateSpanningTrees();
  const softMarginals = new Map();
  const hardTrees = new Map(samples.map((sample) => [sample.id, maximumSpanningTree(sample.demoScores)]));

  let currentRepresentation = "scores";
  let highlightedEdge = null;
  let pinnedEdge = null;
  let replayToken = 0;

  samples.forEach((sample) => {
    renderGraph(`dependency-graph-${sample.id}`, sample);
    renderMatrix(`dependency-matrix-${sample.id}`, sample);
  });

  updateTemperature(Number(slider.value));
  setRepresentation("scores");

  slider.addEventListener("input", () => {
    replayToken += 1;
    showManualSoftState();
    updateTemperature(Number(slider.value));
    updateEpochCounter(1);
    status.textContent = `Soft tree probabilities updated for temperature ${Number(slider.value).toFixed(2)}`;
  });
  replayButton.addEventListener("click", replay);
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
      viewBox: "0 0 360 280",
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
      group.append(hitLine, visualLine);
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
        svgElement("circle", { class: "node-disc", cx: x, cy: y, r: 24 }),
        svgText({ class: "node-symbol", x, y: y + 6, "text-anchor": "middle" }, `z${node}`),
      );
      svg.append(nodeGroup);
    });

    target.replaceChildren(svg);
  }

  function renderMatrix(targetId, sample) {
    const target = figure.querySelector(`#${targetId}`);
    const grid = document.createElement("div");
    grid.className = "score-matrix";
    grid.setAttribute("role", "grid");
    grid.setAttribute("aria-label", `${sample.name} pairwise dependence matrix`);
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
    samples.forEach((sample) => softMarginals.set(sample.id, calculateSoftMarginals(sample.demoScores, temperature)));

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
    figure.dataset.representation = currentRepresentation;
    figure.classList.toggle("is-soft-state", currentRepresentation === "soft");
    figure.classList.toggle("is-hard-state", currentRepresentation === "hard");

    samples.forEach((sample) => {
      updateGraph(sample, currentRepresentation);
      updateMatrix(sample, currentRepresentation);
      const representationLabel = figure.querySelector(`#representation-label-${sample.id}`);
      representationLabel.textContent = currentRepresentation === "scores"
        ? "Pairwise scores"
        : currentRepresentation === "soft"
          ? softProbabilityLabel
          : "Hard MWST";
    });
  }

  function updateGraph(sample, representation) {
    const hardTree = hardTrees.get(sample.id);
    const marginals = softMarginals.get(sample.id);
    figure.querySelectorAll(`#dependency-graph-${sample.id} .graph-edge`).forEach((group) => {
      const edgeId = group.dataset.edge;
      const line = group.querySelector(".edge-line");
      const isHardEdge = hardTree.has(edgeId);
      let strength;

      if (representation === "scores") {
        strength = sample.demoScores[edgeId];
        line.style.stroke = "#216d9c";
        line.style.strokeWidth = `${1.2 + strength * 7}px`;
        line.style.opacity = `${0.16 + strength * 0.84}`;
      } else if (representation === "soft") {
        strength = marginals[edgeId];
        line.style.stroke = "#1b7994";
        line.style.strokeWidth = `${1 + strength * 8}px`;
        line.style.opacity = `${0.1 + strength * 0.9}`;
      } else if (isHardEdge) {
        line.style.stroke = "#0d7d77";
        line.style.strokeWidth = "7px";
        line.style.opacity = "1";
      } else {
        line.style.stroke = "#94a6b5";
        line.style.strokeWidth = "1px";
        line.style.opacity = "0.025";
      }

      line.style.strokeDasharray = "none";
      group.classList.toggle("is-hard-hidden", representation === "hard" && !isHardEdge);
      group.setAttribute("aria-label", edgeAriaLabel(sample, edgeId));
    });
  }

  function updateMatrix(sample, matrixMode) {
    const useSoftProbabilities = matrixMode !== "scores";
    const values = useSoftProbabilities ? softMarginals.get(sample.id) : sample.demoScores;
    const caption = matrixMode === "hard" ? hardMatrixLabel : useSoftProbabilities ? softProbabilityLabel : "Pairwise scores";
    const target = figure.querySelector(`#dependency-matrix-${sample.id}`);
    const captionElement = figure.querySelector(`#matrix-caption-${sample.id}`);
    captionElement.textContent = caption;
    target.querySelectorAll(".matrix-cell").forEach((cell) => {
      const value = values[cell.dataset.edge];
      cell.style.setProperty("--strength", value.toFixed(4));
      cell.setAttribute("aria-label", `${edgeAriaLabel(sample, cell.dataset.edge)}. ${caption}.`);
    });
    target.querySelector(".score-matrix").setAttribute("aria-label", `${sample.name} ${caption.toLowerCase()} matrix`);
  }

  function edgeAriaLabel(sample, edgeId) {
    const edge = edgeById.get(edgeId);
    const hardState = hardTrees.get(sample.id).has(edgeId) ? "selected by the hard MWST" : "not selected by the hard MWST";
    const softProbability = softMarginals.get(sample.id)?.[edgeId] ?? 0;
    return `Illustrative method example. z${edge.a} to z${edge.b}; pairwise score ${sample.demoScores[edgeId].toFixed(2)}; soft inclusion probability ${softProbability.toFixed(2)}; ${hardState}`;
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
    tooltip.textContent = `Illustrative method example\nz${edge.a} - z${edge.b}\n\npairwise score: ${sample.demoScores[edgeId].toFixed(2)}\nsoft P(edge): ${softProbability.toFixed(2)}\nhard MWST: ${hardState}`;
    tooltip.hidden = false;

    const rect = anchor.getBoundingClientRect();
    const tooltipWidth = 210;
    const left = Math.max(12, Math.min(rect.left + rect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - 12));
    const top = Math.max(12, rect.top - 135);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function setSampling(isSampling) {
    figure.classList.toggle("is-sampling", isSampling);
  }

  function showManualSoftState() {
    figure.classList.remove("is-replaying", "is-complete", "is-hard-revealed", "feedback-pulse-active");
    figure.classList.add("is-scores-revealed");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.add("is-visible"));
    setSampling(false);
    setRepresentation("soft");
  }

  function showReadyState() {
    const highTemperature = Number(slider.max);
    figure.classList.add("is-replaying");
    figure.classList.remove("is-complete", "is-scores-revealed", "is-hard-revealed", "feedback-pulse-active");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.remove("is-visible"));
    figure.querySelectorAll('[data-step="inputs"], [data-step="encoder"], [data-step="soft"], [data-step="samples"], [data-step="decoder"], [data-step="recon"]').forEach((element) => element.classList.add("is-visible"));
    slider.value = highTemperature.toFixed(2);
    updateTemperature(highTemperature);
    updateEpochCounter(1);
    setRepresentation("scores");
    slider.disabled = true;
    replayButton.textContent = "Play animation";
    status.textContent = "Ready to animate input-specific dependence learning";
  }

  async function replay() {
    const token = ++replayToken;
    const highTemperature = Number(slider.max);
    const lowTemperature = Number(slider.min) + 0.04;
    const animationDuration = (duration) => reducedMotion.matches ? 0 : duration * 1.65;
    figure.classList.add("is-replaying");
    figure.classList.remove("is-complete", "is-scores-revealed", "is-hard-revealed", "feedback-pulse-active");
    figure.querySelectorAll('[data-step="nodes"], [data-step="fork"], [data-step="feedback"]').forEach((element) => element.classList.remove("is-visible"));
    pinnedEdge = null;
    clearHighlight();
    setSampling(false);
    replayButton.disabled = true;
    slider.disabled = true;
    slider.value = highTemperature.toFixed(2);
    updateTemperature(highTemperature);
    updateEpochCounter(1);
    setRepresentation("scores");

    if (!await pause(animationDuration(320), token)) return;
    if (!await revealStep("nodes", "Revealing latent variables for both inputs", animationDuration(280), token)) return;

    figure.classList.add("is-scores-revealed");
    status.textContent = "Revealing pairwise scores and their matching heatmaps";
    if (!await pause(animationDuration(540), token)) return;
    if (!await revealStep("fork", "Splitting the same pairwise scores into hard and soft uses", animationDuration(300), token)) return;
    if (!await revealStep("soft", "Entering the differentiable soft-tree training surrogate", animationDuration(220), token)) return;

    setRepresentation("soft");
    status.textContent = "Training for 20 epochs as temperature falls";
    if (!await animateEpochs(highTemperature, lowTemperature, token)) return;

    figure.classList.add("feedback-pulse-active");
    if (!await revealStep("feedback", "Sending the differentiable training signal back to the encoder", animationDuration(380), token)) return;
    figure.classList.remove("feedback-pulse-active");

    setRepresentation("hard");
    figure.classList.add("is-hard-revealed");
    status.textContent = "Selecting the hard maximum-weight spanning trees";
    if (!await pause(animationDuration(420), token)) return;

    setSampling(true);
    status.textContent = "Sampling from the hard-tree posterior";
    if (!await pause(animationDuration(640), token)) return;
    setSampling(false);

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
    epochCounter.textContent = `Epoch ${epoch} / 20`;
  }

  function animateEpochs(start, end, token) {
    return new Promise((resolve) => {
      const totalEpochs = 20;
      const epochDuration = 500;
      if (reducedMotion.matches) {
        slider.value = end.toFixed(2);
        updateTemperature(end);
        updateEpochCounter(totalEpochs);
        resolve(token === replayToken);
        return;
      }

      let epoch = 1;
      const tick = () => {
        if (token !== replayToken) {
          resolve(false);
          return;
        }
        const progress = (epoch - 1) / (totalEpochs - 1);
        const temperature = start + (end - start) * progress;
        slider.value = temperature.toFixed(2);
        updateTemperature(temperature);
        updateEpochCounter(epoch);
        if (epoch < totalEpochs) {
          epoch += 1;
          window.setTimeout(tick, epochDuration);
          return;
        }
        window.setTimeout(() => resolve(token === replayToken), epochDuration);
      };
      window.setTimeout(tick, 0);
    });
  }
})();