"use strict";


function populateNodeList() {
    //var arr = ['one', 'two', 'three', 'four', 'five'];
    //arr.forEach( s => {
    //    d3.select("#myDropdown").append().html('<span onclick="spanClicked(this)">' + s + '</span>');
    //});
    d3.json(SERVER_BASE_URL+"getAllNodeNames/")
        .then(json => {
            var nodeList = json['nodeList'];
            var dd = d3.select("#myDropdown");
            nodeList.forEach( n => {
                //dd.append("div").html('<span onclick="spanClicked(this)">' + n + '</span>');
                dd.append("span").attr("onclick", "spanClicked(this)").text(n);
                //dd.append('<span onclick="spanClicked(this)">' + n + '</span>');
            });

            var dl = d3.select("#nodeSuggestions");
            nodeList.forEach( n => {
                dl.append("option").attr("value", n);
            });
        })
        .catch(err => {
            console.error(err);
        });
}
//window.populateNodeList = populateNodeList;

function populateParamLists() {

    d3.json(SERVER_BASE_URL + "getAllParamNames/")
        .then(json => {
            var nodeParams = json['nodeParams'];
            var techParams = json['techParams'];
            // ::NOTE:: You need the `node()` call in these below to extract the actual element itself
            // out of the d3-related container that it comes back in.
            var ddNode = d3.select("#selectNodeParams").node();
            var ddTech = d3.select("#selectTechParams").node();
            nodeParams.forEach( n => {
                //ddNode.append("span").attr("onclick", "console.log('node')").text(n);
                const option = document.createElement('option');
                option.textContent = n;
                option.value = n;
                ddNode.appendChild(option);
            });
            techParams.forEach( n => {
                //ddTech.append("span").attr("onclick", "console.log('tech')").text(n);
                const option = document.createElement('option');
                option.textContent = n;
                option.value = n;
                ddTech.appendChild(option);
            });

        })
        .catch(err => {
            console.error(err);
        });
}



function filterFunction() {
    var input, filter, div, ul, li, span, i;
    input = document.getElementById("dropdownInput");
    filter = input.value.toUpperCase();
    div = document.getElementById("myDropdown");
    span = div.getElementsByTagName("span");
    for(i=0; i < span.length; i++) {
        var txtValue = span[i].textContent || span[i].innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            span[i].style.display = "";
        }else{
            span[i].style.display = "none";
        }
    }
}
window.filterFunction = filterFunction;

function filterFunction_nodeParams() {
    var input, filter, sel, opts, i;
    input = document.getElementById("dropdownInput_nodeParams");
    filter = input.value.toUpperCase();
    sel = document.getElementById("selectNodeParams");
    opts = sel.getElementsByTagName("option");
    for(i=0; i < opts.length; i++) {
        var txtValue = opts[i].value || opts[i].innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            opts[i].style.display = "";
        }else{
            opts[i].style.display = "none";
        }
    }
}
window.filterFunction_nodeParams = filterFunction_nodeParams;

function filterFunction_techParams() {
    var input, filter, sel, opts, i;
    input = document.getElementById("dropdownInput_techParams");
    filter = input.value.toUpperCase();
    sel = document.getElementById("selectTechParams");
    opts = sel.getElementsByTagName("option");
    for(i=0; i < opts.length; i++) {
        var txtValue = opts[i].value || opts[i].innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            opts[i].style.display = "";
        }else{
            opts[i].style.display = "none";
        }
    }
}
window.filterFunction_techParams = filterFunction_techParams;

export {
    populateNodeList,
    populateParamLists,
    filterFunction,
    filterFunction_nodeParams,
    filterFunction_techParams
}
