function displayColors() {
    var hex1;
    var hex2;
    var hex3;
    var clr;
    var steps = [
        '00',
        '33',
        '66',
        '99',
        'cc',
        'ff' 
    ];
    var arrLength = steps.length;
    var counter = 1; // Make sure there are 216 colors displayed
        
        for (var a = 0; a < arrLength; a++) {
            for (var b = 0; b < arrLength; b++) {
                for (var c = 0; c < arrLength; c++) {
                    hex1 = steps[a];
                    hex2 = steps[b];
                    hex3 = steps[c];
                    clr = hex1 + hex2 + hex3;
                    document.getElementById("display").innerHTML += "<div>" + counter + ": " + clr + "</div>";
                    counter++;
                }
            }
        }
}