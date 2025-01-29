$("#show_graph").click(function(){
    var endpoint = '../api/graph/data/';
    var chart_data = [];
    var file_name = document.getElementById("id_system").value;
    $("#system_graph").html("");
    $.ajax({
        method: "GET",
        url: endpoint,
        dataType: "json",
        data: {"file_name":file_name},
        success: function(data){
            chart_data  = data     
            chart = ForceGraph(chart_data, {
                nodeId: d => d.id,
                nodeGroup: d => d.group,
                nodeTitle: d => `${d.id}\n${d.group}\n${d.value}\n${d.value}`,
                nodeValues: d => `${d.id}`,
                // linkStrokeWidth: l => Math.sqrt(l.value),
                linkStrokeWidth: 4,
                width:innerWidth,
                height: innerHeight,
                nodeStrokeWidth:0.1,
                 // a promise to stop the simulation when the cell is re-run
              })
        },
        error: function(error_data){
            console.log("error")
            console.log(error_data)
        }
    });
    
    function ForceGraph({
        nodes, // an iterable of node objects (typically [{id}, …])
        links // an iterable of link objects (typically [{source, target}, …])
    }, {
        nodeId = d => d.id, // given d in nodes, returns a unique identifier (string)
        nodeValues,
        nodeGroup, // given d in nodes, returns an (ordinal) value for color
        nodeGroups, // an array of ordinal values representing the node groups
        nodeTitle, // given d in nodes, a title string
        nodeFill = "currentColor", // node stroke fill (if not using a group color encoding)
        nodeStroke = "#fff", // node stroke color
        nodeStrokeWidth = 1.5, // node stroke width, in pixels
        nodeStrokeOpacity = 1, // node stroke opacity
        nodeRadius = 8, // node radius, in pixels
        nodeStrength,
        linkSource = ({source}) => source, // given d in links, returns a node identifier string
        linkTarget = ({target}) => target, // given d in links, returns a node identifier string
        linkStroke = "black", // link stroke color
        linkStrokeOpacity = 1, // link stroke opacity
        linkStrokeWidth = 1.5, // given d in links, returns a stroke width in pixels
        linkStrokeLinecap = "round", // link stroke linecap
        linkStrength,
        // colors = d3.schemeTableau10, // an array of color strings, for the node groups
        colors = ['green','red'],
        width = 640, // outer width, in pixels
        height = 400, // outer height, in pixels
        invalidation, // when this promise resolves, stop the simulation
    } = {}) {
        // Compute values.
        const N = d3.map(nodes, nodeId).map(intern);
        const LS = d3.map(links, linkSource).map(intern);
        const LT = d3.map(links, linkTarget).map(intern);
        if (nodeTitle === undefined) nodeTitle = (_, i) => N[i];
        const T = nodeTitle == null ? null : d3.map(nodes, nodeTitle);
        const G = nodeGroup == null ? null : d3.map(nodes, nodeGroup).map(intern);
        const W = typeof linkStrokeWidth !== "function" ? null : d3.map(links, linkStrokeWidth);
        const L = typeof linkStroke !== "function" ? null : d3.map(links, linkStroke);
        const K = nodeValues == null ? null : d3.map(nodes, nodeValues).map(intern);
        
        // Replace the input nodes and links with mutable objects for the simulation.
        nodes = d3.map(nodes, (_, i) => ({id: N[i]}));
        links = d3.map(links, (_, i) => ({source: LS[i], target: LT[i]}));
        
        // Compute default domains.
        if (G && nodeGroups === undefined) nodeGroups = d3.sort(G);
        
        // Construct the scales.
        const color = nodeGroup == null ? null : d3.scaleOrdinal(nodeGroups, colors);
        
        // Construct the forces.
        const forceNode = d3.forceManyBody();
        const forceLink = d3.forceLink(links).id(({index: i}) => N[i]);
        if (nodeStrength !== undefined) forceNode.strength(nodeStrength);
        if (linkStrength !== undefined) forceLink.strength(linkStrength);
        
        const simulation = d3.forceSimulation(nodes)
        .force("link", forceLink)
        .force("charge", forceNode)
        .force("center",  d3.forceCenter())
        .on("tick", ticked);
        
        
        const svg = d3.select("#system_graph").append('svg')
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height])
        .attr("style", "max-width: 100%; height: %90;");  
        // .attr("style", "max-width: 100%; height: auto; height: intrinsic;");  

 
        
        const link = svg.append("g")
        .attr("stroke", typeof linkStroke !== "function" ? linkStroke : null)
        .attr("stroke-opacity", linkStrokeOpacity)
        .attr("stroke-width", typeof linkStrokeWidth !== "function" ? linkStrokeWidth : null)
        .attr("stroke-linecap", linkStrokeLinecap)
        .selectAll("line")
        .data(links)
        .join("line");
        
        const node = svg.append("g")
        .attr("fill", nodeFill)
        .attr("stroke", nodeStroke)
        .attr("stroke-opacity", nodeStrokeOpacity)
        .attr("stroke-width", nodeStrokeWidth)
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", nodeRadius)
        .call(drag(simulation));

        const myText = svg.selectAll(".mytext")
			.data(nodes)
			.enter()
			.append("text");
			//the rest of your code
        
        myText.style("fill", "black")
	    .attr("width", "100")
        .attr("height", "100")
        .attr("font-size","20");
        // .text(K);

        if (W) link.attr("stroke-width", ({index: i}) => W[i]);
        if (L) link.attr("stroke", ({index: i}) => L[i]);    
        if (G) node.attr("fill", ({index: i}) => color(G[i]));
        if (T) node.append("title").text(({index: i}) => T[i]);
        if (K) myText.text(({index: i}) => K[i]);
        if (invalidation != null) invalidation.then(() => simulation.stop());
               
        function intern(value) {
            return value !== null && typeof value === "object" ? value.valueOf() : value;
        }


        function ticked() {
            link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
            node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

            myText
            .attr("x", d => d.x + 10)
            .attr("y", d => d.y + 10);
        }
        
        
        function drag(simulation) {    
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.01).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
                // d3.event.sourceEvent.stopPropagation();

                // d.fixed |= 2;
            }
            
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            
            return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
        }

        
        return Object.assign(svg.node(), {scales: {color}});
    }
});


