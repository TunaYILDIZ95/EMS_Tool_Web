$("#id_country").ready(function () {
  const countryId = $("#id_country").children("option").filter(":selected").text();  // get the selected country ID from the HTML input
  const countryName = $("#id_country").children("option").filter(":selected").val();
  const changeStatus = 0; 
  $.ajax({
      url: '/accounts/ajax/load_cities',
      data: {'country_id': countryId, 'country_name':countryName, 'change_status':changeStatus},    // add the country id to the GET parameters
      success: function(data){
          //$("#id_city").html(data);
          let html_data = '<option value="">---------</option>';
          data.cities.forEach(function (city) {
              html_data += `<option value="${city}">${city}</option>`             
          });
          $("#id_city").html(html_data);
          $("#id_city").val(data.user_data);
      }
  });
  
});
$("#id_country").change(function () {
  const countryId = $(this).find(":selected").text();  // get the selected country ID from the HTML input
  const countryName = $(this).find(":selected").val();
  const changeStatus = 1;
  $.ajax({
      url: '/accounts/ajax/load_cities',   
      data: {'country_id': countryId, 'country_name':countryName, 'change_status':changeStatus},  // add the country id to the GET parameters     
      success: function(data){
          //$("#id_city").html(data);
          let html_data = '<option value="">---------</option>';
          data.cities.forEach(function (city) {
              html_data += `<option value="${city}">${city}</option>`
          });
          $("#id_city").html(html_data);
      }
  });
  
});