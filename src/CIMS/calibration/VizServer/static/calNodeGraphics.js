"use strict";



// These functions are mostly straight suggestions from Lumo on 2026-07-06 17:56:36

function drawMiniBars(parentG, data, cw, ch, color){

    if (!data.length) return;

    const x = d3.scaleBand()
        .domain(data.map((_,i) => i))
        .range([-cw / 2 + 2, cw / 2 - 2])
        .padding(0.2);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data)])
        .range([ch / 2 - 2, -ch / 2 + 2]);  // inverted (SVG y grows downward)

    parentG.append("g")
        .attr("class", "mini-bars")
        .selectAll("rect.bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", (_,i) => x(i))
        .attr("width", x.bandwidth())
        .attr("y", d => y(d))
        .attr("height", d => (ch / 2 - 2) - y(d))
        .attr("fill", color)
        .attr("opacity", 0.85);
}


function drawMiniPie(parentG, data, color) {

    if (!data.length) return;

    const radius = 16;

    const palette = d3.scaleOrdinal(
        d3.schemeCategory10
    );

    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);

    const pie = d3.pie();

    parentG.append("g")
        .attr("class", "mini-pie")
        .attr("transform", "translate(0,0)")
        .selectAll("path.slice")
        .data(pie(data))
        .enter()
        .append("path")
        .attr("class", "slice")
        .attr("d", arc)
        .attr("fill", (_,i) => palette(i));

}

// Sparkline alternative — one path per node
function drawSparkline(parentG, data, cw, ch, color) {
  if (data.length < 2) return;
  const line = d3.line()
    .x((_, i) => (i / (data.length - 1)) * (cw - 4) - (cw - 4) / 2)
    .y(v => (ch / 2 - 2) - (v / d3.max(data)) * (ch - 4));
  parentG.append("path")
    .attr("d", line(data))
    .attr("fill", "none")
    .attr("stroke", color)
    .attr("stroke-width", 1.5);
}

export {
    drawMiniBars,
    drawMiniPie,
    drawSparkline
}
