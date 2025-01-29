$("#id_system").change(function(){
    var endpoint = '../api/pf_solution/data/';
    var table_data = [];
    var file_name = document.getElementById("id_system").value;

    $("#informationTableContent").html("");

    $.ajax({
        method: "GET",
        url: endpoint,
        dataType: "json",
        data: {"file_name":file_name},
        success: function(data){
            table_data = data
            var phase = 1;
            var counter = 0;
            for (let i=0; i < table_data['voltages'].length; i++){
                $('#informationTable').find('tbody').append("<tr><th>"+i+"</th><td>"+table_data["bus_list"][counter]+"</td><td>"
                +phase+"</td><td>"+Number(table_data["voltages"][i]).toFixed(4)+"</td><td>"+Number(table_data["angles"][i]).toFixed(2)+"</td></tr>");
                
                if(phase<3){
                    phase += 1;
                }
                else{
                    phase = 1;
                    counter += 1;
                }
            }
        },
        error: function(error_data){
            console.log("error")
            console.log(error_data)
        }
    });

})