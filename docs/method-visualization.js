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
  const nodePositions = {
    1: [132, 24],
    2: [236, 86],
    3: [186, 178],
    4: [48, 134],
  };
  const sampleNodePositions = {
    1: [66, 26],
    2: [110, 72],
    3: [85, 153],
    4: [20, 116],
  };

  // Deliberately schematic scores make the two explanatory topologies unambiguous.
  const samples = [
    {
      id: "a",
      name: "Input A",
      demoScores: { 12: 0.94, 13: 0.88, 14: 0.84, 23: 0.31, 24: 0.22, 34: 0.27 },
      values: { 1: "+0.7", 2: "-0.2", 3: "+1.1", 4: "+0.1" },
    },
    {
      id: "b",
      name: "Input B",
      demoScores: { 12: 0.88, 13: 0.3, 14: 0.17, 23: 0.92, 24: 0.35, 34: 0.86 },
      values: { 1: "-0.5", 2: "+0.3", 3: "+0.9", 4: "-0.4" },
    },
  ];

  const slider = figure.querySelector("#tree-temperature");
  const temperatureValue = figure.querySelector("#temperature-value");
  const replayButton = figure.querySelector("#method-replay");
  const tooltip = figure.querySelector("#edge-tooltip");
  const status = figure.querySelector("#method-status");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const allTrees = enumerateSpanningTrees();
  const softMarginals = new Map();
  const hardTrees = new Map(samples.map((sample) => [sample.id, maximumSpanningTree(sample.demoScores)]));

  let highlightedEdge = null;
  let pinnedEdge = null;
  let replayToken = 0;

  samples.forEach((sample) => {
    renderGraph(`score-graph-${sample.id}`, sample, "score");
    renderScoreMatrix(`score-matrix-${sample.id}`, sample);
    renderGraph(`hard-graph-${sample.id}`, sample, "hard");
    renderGraph(`soft-graph-${sample.id}`, sample, "soft");
    renderGraph(`sample-graph-${sample.id}`, sample, "sample");
  });

  updateTemperature(Number(slider.value));
  slider.addEventListener("input", () => {
    replayToken += 1;
    updateTemperature(Number(slider.value));
    figure.classList.add("has-manual-temperature");
  });
  replayButton.addEventListener("click", replay);

  figure.classList.add("is-animated");
  window.setTimeout(() => {
    replay();
  }, 0);

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

  function renderGraph(targetId, sample, kind) {
    const target = figure.querySelector(`#${targetId}`);
    const isSample = kind === "sample";
    const positions = isSample ? sampleNodePositions : nodePositions;
    const svg = svgElement("svg", {
      class: `latent-graph latent-graph--${kind}`,
      viewBox: isSample ? "0 0 132 182" : "0 0 284 204",
      role: "img",
      "aria-label": `${sample.name} ${kind} latent graph`,
    });
    const hardTree = hardTrees.get(sample.id);

    edges.forEach((edge, edgeIndex) => {
      if (kind === "sample" && !hardTree.has(edge.id)) {
        return;
      }

      const group = svgElement("g", {
        class: `graph-edge graph-edge--${kind}`,
        "data-edge": edge.id,
        "data-sample": sample.id,
        tabindex: "0",
        role: "button",
        "aria-label": edgeAriaLabel(sample, edge.id, kind),
        style: `--edge-index:${edgeIndex}`,
      });
      if (kind === "hard" || kind === "sample") {
        group.classList.add(hardTree.has(edge.id) ? "is-hard-selected" : "is-hard-unselected");
      }

      const [startX, startY] = positions[edge.a];
      const [endX, endY] = positions[edge.b];
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

      if (kind === "score") {
        const score = sample.demoScores[edge.id];
        visualLine.style.strokeWidth = `${1.1 + score * 6.4}px`;
        visualLine.style.opacity = `${0.18 + score * 0.82}`;
      }
      group.append(hitLine, visualLine);
      attachEdgeInteractions(group, sample.id, edge.id);
      svg.append(group);
    });

    nodes.forEach((node) => {
      const [x, y] = positions[node];
      const nodeGroup = svgElement("g", {
        class: "graph-node",
        "data-node": node,
        "data-sample": sample.id,
      });
      const radius = isSample ? 25 : 18;
      nodeGroup.append(
        svgElement("circle", { class: "node-disc", cx: x, cy: y, r: radius }),
        svgText("text", { class: "node-symbol", x, y: kind === "sample" ? y - 5 : y + 4, "text-anchor": "middle" }, `z${node}`),
      );
      if (kind === "sample") {
        nodeGroup.append(
          svgText("text", { class: "node-value", x, y: y + 13, "text-anchor": "middle" }, sample.values[node]),
        );
      }
      svg.append(nodeGroup);
    });

    target.replaceChildren(svg);
  }

  function svgText(name, attributes, text) {
    const element = svgElement(name, attributes);
    element.textContent = text;
    return element;
  }

  function renderScoreMatrix(targetId, sample) {
    const target = figure.querySelector(`#${targetId}`);
    const grid = document.createElement("div");
    grid.className = "score-matrix";
    grid.setAttribute("role", "grid");
    grid.setAttribute("aria-label", `${sample.name} symmetric pairwise score matrix`);
    grid.append(matrixLabel("", "matrix-corner"));
    nodes.forEach((node) => grid.append(matrixLabel(`z${node}`, "matrix-axis", "columnheader")));

    nodes.forEach((rowNode) => {
      grid.append(matrixLabel(`z${rowNode}`, "matrix-axis", "rowheader"));
      nodes.forEach((columnNode) => {
        if (rowNode === columnNode) {
          grid.append(matrixLabel("", "matrix-diagonal", "gridcell"));
          return;
        }

        const edge = edgeIdFor(rowNode, columnNode);
        const score = sample.demoScores[edge];
        const button = document.createElement("button");
        button.type = "button";
        button.className = "matrix-cell";
        button.dataset.edge = edge;
        button.dataset.sample = sample.id;
        button.style.setProperty("--score", score.toFixed(2));
        button.style.setProperty("--edge-index", String(edges.findIndex((candidate) => candidate.id === edge)));
        button.setAttribute("role", "gridcell");
        button.setAttribute("aria-label", edgeAriaLabel(sample, edge, "score matrix"));
        button.textContent = score.toFixed(2);
        attachEdgeInteractions(button, sample.id, edge);
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

  function edgeAriaLabel(sample, edgeId, representation) {
    const edge = edgeById.get(edgeId);
    const hardState = hardTrees.get(sample.id).has(edgeId) ? "selected by the hard MWST" : "not selected by the hard MWST";
    return `z${edge.a} to z${edge.b}, pairwise score ${sample.demoScores[edgeId].toFixed(2)}, ${hardState}, ${representation}`;
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
    tooltip.textContent = `z${edge.a} - z${edge.b}\n\npairwise score: ${sample.demoScores[edgeId].toFixed(2)}\nsoft P(edge): ${softProbability.toFixed(2)}\nhard MWST: ${hardState}`;
    tooltip.hidden = false;

    const rect = anchor.getBoundingClientRect();
    const tooltipWidth = 190;
    const left = Math.max(12, Math.min(rect.left + rect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - 12));
    const top = Math.max(12, rect.top - 118);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function updateTemperature(temperature) {
    temperatureValue.value = temperature.toFixed(2);
    temperatureValue.textContent = temperature.toFixed(2);
    slider.setAttribute("aria-valuetext", `${temperature.toFixed(2)} temperature`);

    samples.forEach((sample) => {
      const marginals = calculateSoftMarginals(sample.demoScores, temperature);
      softMarginals.set(sample.id, marginals);
      const softGraph = figure.querySelector(`#soft-graph-${sample.id}`);
      softGraph.querySelectorAll(".graph-edge").forEach((group) => {
        const probability = marginals[group.dataset.edge];
        const line = group.querySelector(".edge-line");
        line.style.strokeWidth = `${1 + probability * 7}px`;
        line.style.opacity = `${0.15 + probability * 0.85}`;
        group.setAttribute("aria-label", `${edgeAriaLabel(sample, group.dataset.edge, "soft tree marginal")}, inclusion probability ${probability.toFixed(2)}`);
      });
    });

    if (highlightedEdge) {
      showTooltip(highlightedEdge.sampleId, highlightedEdge.edgeId, highlightedEdge.anchor);
    }
  }

  async function replay() {
    const token = ++replayToken;
    const highTemperature = Number(slider.max);
    const lowTemperature = Number(slider.min) + 0.04;
    figure.classList.remove("has-manual-temperature");
    figure.classList.remove("is-complete");
    figure.classList.add("is-replaying");
    figure.classList.remove("feedback-pulse-active");
    figure.querySelectorAll("[data-step]").forEach((element) => element.classList.remove("is-visible"));
    pinnedEdge = null;
    clearHighlight();
    updateTemperature(highTemperature);

    if (!await revealStep("inputs", "Showing two inputs", 330, token)) return;
    if (!await revealStep("encoder", "Shared encoder predicts pairwise dependence", 330, token)) return;
    if (!await revealStep("scores", "Revealing candidate latent graphs and score matrices", 620, token)) return;
    if (!await revealStep("hard", "Selecting the hard maximum-weight spanning trees", 440, token)) return;
    if (!await revealStep("soft", "Showing soft tree marginals used for training", 300, token)) return;

    status.textContent = "Sharpening the soft tree distribution";
    const completedTemperature = await animateTemperature(highTemperature, lowTemperature, reducedMotion.matches ? 0 : 1100, token);
    if (!completedTemperature) return;

    figure.classList.add("feedback-pulse-active");
    if (!await revealStep("feedback", "Sending the differentiable training signal back to the encoder", 430, token)) return;
    figure.classList.remove("feedback-pulse-active");
    if (!await revealStep("samples", "Sampling correlated latent values from the hard trees", 380, token)) return;
    if (!await revealStep("decoder", "Passing samples through the shared decoder", 300, token)) return;
    if (!await revealStep("recon", "Showing schematic reconstructions", 260, token)) return;

    figure.classList.remove("is-replaying");
  figure.classList.add("is-complete");
    status.textContent = "Interactive method visualization ready";
  }

  async function revealStep(step, message, duration, token) {
    if (token !== replayToken) {
      return false;
    }
    figure.querySelectorAll(`[data-step="${step}"]`).forEach((element) => element.classList.add("is-visible"));
    status.textContent = message;
    if (reducedMotion.matches || duration === 0) {
      return token === replayToken;
    }
    await new Promise((resolve) => window.setTimeout(resolve, duration));
    return token === replayToken;
  }

  function animateTemperature(start, end, duration, token) {
    return new Promise((resolve) => {
      if (duration === 0) {
        slider.value = String(end);
        updateTemperature(end);
        resolve(token === replayToken);
        return;
      }
      const startedAt = window.performance.now();
      const tick = () => {
        if (token !== replayToken) {
          resolve(false);
          return;
        }
        const now = window.performance.now();
        const progress = Math.min((now - startedAt) / duration, 1);
        const easedProgress = 1 - (1 - progress) * (1 - progress);
        const temperature = start + (end - start) * easedProgress;
        slider.value = temperature.toFixed(2);
        updateTemperature(temperature);
        if (progress < 1) {
          window.setTimeout(tick, 16);
          return;
        }
        resolve(true);
      };
      window.setTimeout(tick, 0);
    });
  }
})();