// Javascript for Pokemon

function main(){

  initMap();
}

function initMap() {

  var mapCenter = new google.maps.LatLng(37.615223, -122.389977);
  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 10,
    center: mapCenter,
    mapTypeId: 'terrain'
  });

  var dragonairImage = {
     url: 'http://pre00.deviantart.net/73f7/th/pre/i/2013/024/f/5/dragonair_by_darkheroic-d5sizqi.png',
     size: new google.maps.Size(70,70),
     origin: new google.maps.Point(0, 0),
     anchor: new google.maps.Point(0, 0),
     scaledSize: new google.maps.Size(60, 60),
     labelOrigin: new google.maps.Point(9, 8)
  };

    // Shapes define the clickable region of the icon. The type defines an HTML
    // <area> element 'poly' which traces out a polygon as a series of X,Y points.
    // The final coordinate closes the poly by connecting to the first coordinate.
    var shape = {
        coords: [0, 0, 0, 50, 50, 50, 50, 0],
        type: 'poly'
    };

    var pok_list = [
        ['Dragonair', 37.5320567008, -122.19265261, 0],
        ['Dragonair', 37.8986906754, -122.29733085, 1],
        ['Dragonair', 37.8600646907, -122.485597556, 2],
        ['Dragonair', 37.8067997243, -122.42385277, 3],
    ];


    // Markers :
    for (var i = 0; i < pok_list.length; i++) {
        var mark = pok_list[i];
        var marker = new google.maps.Marker({
            position: {lat: mark[1], lng: mark[2]},
            map: map,
            icon: window.top.dragonairImage,
            shape: shape,
            title: mark[0] + " : " + mark[3],
            zIndex: mark[3],
            label: {
                text: i.toString(),
                fontWeight: 'bold',
                fontSize: '40px',
                fontFamily: '"Courier New", Courier,Monospace',
                color: 'black'
            }
        });
    }


    var bounds = new google.maps.LatLngBounds; // automate bounds
    var dist = '';

    var outputDiv = document.getElementById('output');
    outputDiv.innerHTML = 'From ----> To ----> distance <br>';

    // Distances :

    function calcDistance(origin1,destinationB,ref_Callback_calcDistance, k, n){
    var service = new google.maps.DistanceMatrixService();
    var temp_duration = 0;
    var temp_distance = 0;
    var testres;
    service.getDistanceMatrix(
            {
            origins: [origin1],
            destinations: [destinationB],
            travelMode: google.maps.TravelMode.DRIVING,
            unitSystem: google.maps.UnitSystem.METRIC,
            avoidHighways: false,
            avoidTolls: false

            }, function(response, status) {
            if (status !== google.maps.DistanceMatrixStatus.OK) {
                alert('Error was: ' + status);
                testres= {"duration":0,"distance":0};

            } else {
                var originList = response.originAddresses;
                var destinationList = response.destinationAddresses;
                var showGeocodedAddressOnMap = function (asDestination) {
                    testres = function (results, status) {
                      if (status === 'OK') {
                        map.fitBounds(bounds.extend(results[0].geometry.location));
                      } else {
                        alert('Geocode was not successful due to: ' + status);
                      }
                    };
                };

                for (var i = 0; i < originList.length; i++) {
                    var results = response.rows[i].elements;
                    for (var j = 0; j < results.length; j++) {
                        temp_duration+=results[j].duration.text;
                        temp_distance+=results[j].distance.text;
                    }
                }
                    testres=[temp_duration,temp_distance];

                    if(typeof ref_Callback_calcDistance === 'function'){
                         //calling the callback function
                        ref_Callback_calcDistance(testres, k, n)

                    }
                }
            }
            );
    }

    function Callback_calcDistance(testres, k, n) {
        dist = testres[1];
        outputDiv.innerHTML += k + ' ----> ' + n + ' ----> ' + dist + '<br>'

        //console.log(testres[1]);
    }

    for (var k = 0; k < pok_list.length; k++) {
        var origin = new google.maps.LatLng(pok_list[k][1], pok_list[k][2]);

        for (var n = 0; n < pok_list.length; n++) {
            if (n !== k) {
                var dest = new google.maps.LatLng(pok_list[n][1],pok_list[n][2]);

                //calling the calcDistance function and passing callback function reference
                calcDistance(origin, dest, Callback_calcDistance, k,n);
            }
        }
    }

}

