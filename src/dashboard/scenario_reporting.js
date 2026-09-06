// Presentation diagnostics only. The propagation recurrence is unchanged.
((root) => {
  const clip = value => Math.max(0, Math.min(100, value));
  const calculate = (data, name, multiplier) => {
    let current = data.banks.map(bank => clip(data.shocks[name][bank] * multiplier));
    const paths = [current.slice()];
    for (let step = 1; step <= 5; step += 1) {
      current = current.map((value, target) => {
        const propagated = current.reduce((total, sourceValue, source) => total + data.adjacency[source][target] * sourceValue, 0);
        return clip(0.70 * value + 0.45 * propagated);
      });
      paths.push(current.slice());
    }
    return paths;
  };
  const leaders = (banks, values) => {
    const peak = Math.max(...values);
    return {
      exact: banks.filter((_, i) => Math.abs(values[i] - peak) <= 1e-9),
      displayed: banks.filter((_, i) => values[i].toFixed(1) === peak.toFixed(1)),
      saturated: banks.filter((_, i) => values[i] >= 100 - 1e-9),
    };
  };
  const summarize = (data, paths, name, severity) => {
    const opening = paths[0], closing = paths[paths.length - 1];
    const initial = leaders(data.banks, opening), final = leaders(data.banks, closing);
    const delta = paths[1].map((value, i) => value - opening[i]);
    const rose = delta.filter(x => x > 1e-9).length;
    const fell = delta.filter(x => x < -1e-9).length;
    const maxMove = (from, to) => Math.max(...to.map((v, i) => Math.abs(v - from[i])));
    const lastMove = maxMove(paths[paths.length - 2], closing);
    const tie = final.displayed.length > 1;
    const bankLabel = (tie ? 'Tie at 0.1 precision: ' : '') + final.displayed.join(', ');
    const saturation = final.saturated.length ? ` Saturation: ${final.saturated.length} of ${data.banks.length} banks at the 100 cap; capped values cannot establish differentiation.` : '';
    const exactNote = tie ? ` Exact maximum: ${final.exact.join(', ')}; values sharing the displayed maximum are treated as tied, not uniquely ranked.` : '';
    return {
      bankLabel,
      title: `${name} (${severity}%): Contagion Propagation`,
      finalTitle: `${name} (${severity}%): Final Stress`,
      response: `${name} at ${severity}% severity: ${tie ? 'no unique terminal leader at displayed precision' : `${final.displayed[0]} has the highest terminal stress`}.${saturation}${exactNote} This is scenario research evidence, not a trade instruction. Scenario selection does not rerun the optimizer.`,
      walkthrough: [
        `Initial highest assumed stress: ${initial.displayed.join(', ')} at ${Math.max(...opening).toFixed(1)}/100. Initial range ${Math.min(...opening).toFixed(1)}–${Math.max(...opening).toFixed(1)}. These magnitudes are preset assumptions.`,
        `After step one, ${rose} banks rise, ${fell} decline and ${data.banks.length - rose - fell} are unchanged. Net changes range ${Math.min(...delta).toFixed(1)} to ${Math.max(...delta).toFixed(1)} points, including persistence and incoming spillover; net change is not spillover alone.`,
        `The run ends after five abstract steps. Largest absolute final-step change: ${lastMove.toFixed(1)} points.${saturation || ' The finite horizon does not establish convergence.'} Steps have no calibrated real-world duration.`,
      ],
      leaders: final,
    };
  };
  const api = {calculate, leaders, summarize};
  root.NorthernScenario = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
