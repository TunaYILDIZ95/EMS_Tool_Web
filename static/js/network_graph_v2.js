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
            chart = ForceGraph(chart_data,{
                //width:innerWidth,
                //height: innerHeight,
            })
        },
        error: function(error_data){
            console.log("error deneme")
            console.log(error_data)
        }
    });
    
    function ForceGraph({
        nodes, // an iterable of node objects (typically [{id}, …])
        links, // an iterable of link objects (typically [{source, target}, …])
        powerSystemData = chart_data
    }, {
        width = 1200, // outer width, in pixels
        height = 1000,
    }= {}) {
        /*const powerSystemData = {
            nodes: [
                { id: 'Bus 1', voltage: 1.0, angle: 0.0},
                { id: 'Bus 2', voltage: 0.95, angle: -2.0},
                { id: 'Bus 3', voltage: 0.98, angle: -4.0},
                { id: 'Bus 4', voltage: 1.01, angle: 2.0},
            ],
            links: [
                { source: 'Bus 1', target: 'Bus 2' , activeFlow :"1 p.u", reactiveFlow :"0.5 p.u"},
                { source: 'Bus 1', target: 'Bus 3' , activeFlow :"2 p.u", reactiveFlow :"1 p.u"},
                { source: 'Bus 1', target: 'Bus 4' , activeFlow :"3 p.u", reactiveFlow :"1.5 p.u"},
            ],
        };*/
        // Compute values.
        const svg = d3.select("#system_graph").append('svg')
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height])
        .attr("style", "max-width: 100%; height: %90;");  
        
        const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(powerSystemData.links)
        .enter().append("line")
        .attr("stroke-width",4)
        .attr("stroke","grey")
        .on("click", (event, d) => displayLinkData(event, d));
        
        function displayLinkData(event, d) {
            // Remove any existing info box
            d3.select(".info-box").remove();
            
            // Create an info box with node data
            const infoBox = d3.select("body")
            .append("div")
            .attr("class", "info-box")
            .style("position", "absolute")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .style("background", "#FFF")
            .style("padding", "8px")
            .style("border", "1px solid #888")
            .style("border-radius", "5px");
            
            // Add node data to the info box
            infoBox.html(`<p><strong>ActiveFlow:</strong> ${d.activeFlow}<br>
            <strong>ReactiveFlow:</strong> ${d.reactiveFlow}</p>`);
        }
        
        const nodeGroup = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(powerSystemData.nodes)
        .enter().append("g")
        .attr("transform", d => `translate(${d.x}, ${d.y})`)
        .on("click", (event, d) => displayNodeData(event, d))
        .call(d3.drag()
        .on("start", (event, d) => dragstarted(event, d))
        .on("drag", (event, d) => dragged(event, d))
        .on("end", (event, d) => dragended(event, d)));
        
        nodeGroup.append("rect")
        .attr("class", "node")
        .attr("width", 10)
        .attr("height", 30)
        .attr("fill", "steelblue")
        .attr("x", -10 / 2)
        .attr("y", -30 / 2);
        
        nodeGroup.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("dy", "-1.4em")
        .text(d => d.id);
        
        function displayNodeData(event, d) {
            // Remove any existing info box
            d3.select(".info-box").remove();
            
            // Create an info box with node data
            const infoBox = d3.select("body")
            .append("div")
            .attr("class", "info-box")
            .style("position", "absolute")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .style("background", "#FFF")
            .style("padding", "8px")
            .style("border", "1px solid #888")
            .style("border-radius", "5px");
            
            // Add node data to the info box
            infoBox.html(`<p><strong>Voltage:</strong> ${d.voltage}<br>
            <strong>Angle:</strong> ${d.angle}</p>`);
        }
        
        // Function to remove the info box when clicking outside the nodes
        d3.select("body").on("click", function(event) {
            if (!event.target.closest(".node")) {
                d3.select(".info-box").remove();
            }
        }, true);      
        
        const simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-5))
        .force("center", d3.forceCenter());
        
        simulation.nodes(powerSystemData.nodes);
        simulation.force("link").links(powerSystemData.links);
        simulation.on("tick", ticked);
        
        
        // Update the positions of nodes, links, and labels
        function ticked() {
            link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
            nodeGroup
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
        }
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = event.x;
            d.fy = event.y;
        }
        
    }
});


