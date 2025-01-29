$("#show_graph").click(function(){
    var endpoint = '../api/graph/data/';
    var chart_data = [];
    $.ajax({
        method: "GET",
        url: endpoint,
        dataType: "json",
        success: function(data){
            // console.log(data)
            chart_data = data;
            console.log(chart_data.nodes)
            createChart(chart_data.nodes, 'system_graph', 'forceDirectedGraph', 'vertical');
            // createChart(chart_data.nodes, 'system_graph', 'dendogram', 'horizontal');
        },
        error: function(error_data){
            console.log("error")
            console.log(error_data)
        }
    });
    
    function createChart(nodes, id, type, orientation) {  
        // const pointImage  = new Image()
        // pointImage.src = 'https://www.chartjs.org/docs/3.5.1/favicon.ico';   
        new Chart(document.getElementById(id).getContext("2d"), {
            type,
            data: {
                labels: nodes.map((d) => d.name),
                datasets: [{
                    pointBackgroundColor: 'steelblue',
                    pointRadius: 10,
                    data: nodes.map((d) => Object.assign({}, d)),
                    // pointStyle : pointImage,
                }]
            },
            
            options: {
                dragData: true,
                dragX: true,
                tree: {
                    orientation
                },
                layout: {
                    padding: {
                        top: 50,
                        left: 15,
                        right: 40,
                        bottom: 20
                    }
                },
                plugins: {
                    legend: {
                        display: false,
                    },
                    datalabels: {
                        align: orientation === 'vertical' ? 'bottom' : 'right',
                        offset: 6,
                        backgroundColor: 'white',
                        formatter: (v) => {
                            return v.name;
                        }
                    },
                    tooltip: {
                        usePointStyle: true,
                    }
                },
            }
        });
    }
    
    
});


