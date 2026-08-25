"use strict";

import { tableToObject } from './tableToObject.js';

function closeNodeData(){
    d3.select("#nodeInfo").html(null);
}
window.closeNodeData = closeNodeData;

function clearPlots(){
    d3.select("#calibrationPlots").html(null);
}
window.clearPlots = clearPlots;

function submitFICs(nodeName){
    console.log("submitFICs method firing with nodeName: ", nodeName);
    var table = d3.select("#nodeInfo table").node();
    var parsedData = tableToObject(table);
    //console.log("we got object: " + JSON.stringify(parsedData));

    // Ok, now the fun part, we need to figure out how to do a POST using this d3 wrapper around `fetch`, providing this
    // `parsedData` object as JSON, with all the mediatype thingies set properly

    var queryURL = new URL(SERVER_BASE_URL + 'setNodeFICs/'+nodeName);

    // The below is also from Lumo...
    d3.json(queryURL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(parsedData)
    })
    .then(response => {
        console.log("Server replied: ", response);
    })
    .catch(error => {
        console.error("Something bad happened...: ", error);
    });

}
window.submitFICs = submitFICs;

function rerunCIMS(nodeName){
    console.log("rerunCIMS method firing with nodeName: ", nodeName);

    var queryURL = new URL(SERVER_BASE_URL + 'rerunCIMS/'+nodeName);
    d3.json(queryURL)
        .then(jsonData => {
            console.log(JSON.stringify(jsonData));
        })
        .catch(error => {
            console.error(error);
        });

}
window.rerunCIMS = rerunCIMS;

function autoCalibrate(nodeName){
    console.log("autoCalibrate method firing with nodeName: ", nodeName);
    var queryURL = new URL(SERVER_BASE_URL + 'autoCalibrate/'+nodeName);
    d3.json(queryURL)
        .then(jsonData => {
            console.log(JSON.stringify(jsonData));
        })
        .catch(error => {
            console.error(error);
        });
}
window.autoCalibrate = autoCalibrate;

function makeMSPlots(nodeName, plotType="stack"){
    console.log("makeMSPlots method firing with nodeName: ", nodeName);
    if(plotType === "stack"){
        var queryURL = new URL(SERVER_BASE_URL + 'ms_plotting_stack/'+nodeName);
    }else if(plotType === "line"){
        var queryURL = new URL(SERVER_BASE_URL + 'ms_plotting_line/'+nodeName);
    }else{
        console.error("Unknown plotType");
    }
    d3.html(queryURL)
        .then(html => {
            console.log("Getting html from makeMSPlots method: " + html);
            d3.select("#calibrationPlots").append("div").html(html.body.innerHTML);
        })
        .catch(error => {
            console.error(error);
        });
}
window.makeMSPlots = makeMSPlots;

function fetchCalibrationInfoForNode(nodeName){
    console.log("Doing Calibration on node: " + nodeName);

    var queryURL = new URL(SERVER_BASE_URL + 'getNodeFICsHTML/'+nodeName)

    d3.html(queryURL)
        .then(html => {
            d3.select("#nodeInfo").html(null);
            d3.select("#nodeInfo").html(html.body.innerHTML);
            d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Close</button>');
            d3.select("#nodeInfo").append().html('<button onclick="clearPlots();">Clear Plots</button>');
            let html_submit = `<button onclick='submitFICs(\"${nodeName}\");'>Submit</button>`;
            let html_rerun = `<button onclick='rerunCIMS(\"${nodeName}\");'>ReRun CIMS</button>`;
            let html_auto = `<button onclick='autoCalibrate(\"${nodeName}\");'>AutoCal</button>`;
            let html_plot = `<button onclick='makeMSPlots(\"${nodeName}\");'>Plot Stack</button>`;
            let html_plot2 = `<button onclick='makeMSPlots(\"${nodeName}\", plotType=\"line\");'>Plot Lines</button>`;
            //console.log("wtf: ", localHTML);
            d3.select("#nodeInfo").append().html(html_submit);
            d3.select("#nodeInfo").append().html(html_rerun);
            d3.select("#nodeInfo").append().html(html_auto);
            d3.select("#nodeInfo").append().html(html_plot);
            d3.select("#nodeInfo").append().html(html_plot2);
        })
        .catch(error => {
            console.error(error);
        });

}


// This is just to get the full node name/address stored in a variable to which the marimo nodebook in 
// which this is (or can be) embedded has access.
function setNodeSelection(nodeName){

    //console.log("Setting selection to node: " + nodeName);
    let queryURL = new URL(SERVER_BASE_URL + 'setNodeSelection/'+nodeName);
    d3.json(queryURL)
        .then(html => {
            //d3.select("#nodeInfo").html(null);
            //d3.select("#nodeInfo").html(html.body.innerHTML);
            //d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Clear</button>');
            //console.log("Selection set");
            //console.log(JSON.stringify(html));
        })
        .catch(error => {
            console.error(error);
        });
}

function fetchNodeData(nodeName){

    console.log("Requested Node Name: " + nodeName);

    // Here we need to look in the node param selector, and see if there are any selections made. If there are
    // not, we want to return all the parameters, which is what will be done by default using this `getServiceTable`
    // endpoint. If there ARE any selections made, we pass those into the GET request as search params, and only
    // those params come back.

    if(hasSelection('selectNodeParams')){

        var pList = getSelectedList('selectNodeParams');
        const params = {params: pList}
        var queryURL = new URL(SERVER_BASE_URL + 'getServiceTable_filtParam/'+nodeName);
        queryURL.search = new URLSearchParams(params);

    }else{
        var queryURL = new URL(SERVER_BASE_URL + 'getServiceTable/'+nodeName);
    }

    

    d3.html(queryURL)
        .then(html => {
            
            d3.select("#nodeInfo").html(null);
            d3.select("#nodeInfo").html(html.body.innerHTML);    
            d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Clear</button>');

        })
        .catch(error => {
            console.error(error);
        });
}


function fetchTechData(nodeName){

    console.log("Requested Tech Node Name: " + nodeName);

    // Same thing as above, but we need to look in the tech params selector.

    if(hasSelection('selectTechParams')){
        var pList = getSelectedList('selectTechParams');
        const params = {params: pList}
        var queryURL = new URL(SERVER_BASE_URL + 'getTechTable_filtParam/'+nodeName);
        queryURL.search = new URLSearchParams(params);
    }else{
        var queryURL = new URL(SERVER_BASE_URL + 'getTechTable/'+nodeName);
    }

    d3.html(queryURL)
        .then(html => {
            
            d3.select("#nodeInfo").html(null);
            d3.select("#nodeInfo").html(html.body.innerHTML);    
            d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Clear</button>');

        })
        .catch(error => {
            console.error(error);
        });
}

function fetchEmissionsData(nodeName){

    console.log("Can we see the server address in here: " + SERVER_BASE_URL);
    let queryURL = new URL(SERVER_BASE_URL + 'getEmissionsTable/'+nodeName);
    d3.html(queryURL)
        .then(html => {
            d3.select("#nodeInfo").html(null);
            d3.select("#nodeInfo").html(html.body.innerHTML);
            d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Clear</button>');
        })
        .catch(error => {
            console.error(error);
        });
}

function fetchRequestedQuantitiesData(nodeName){

    let queryURL = new URL(SERVER_BASE_URL + 'getRequestedQuantitiesTable/'+nodeName);
    d3.html(queryURL)
        .then(html => {
            d3.select("#nodeInfo").html(null);
            d3.select("#nodeInfo").html(html.body.innerHTML);
            d3.select("#nodeInfo").append().html('<button onclick="closeNodeData();">Clear</button>');
        })
        .catch(error => {
            console.error(error);
        });
}


function makeReqProvLinks(root) {

    var missingNodeNames = [];

    var reqProvLinks = graphInput.links.map( ll => {
        let src = root.descendants().reverse().filter( d => d.data.id === ll.source );
        let targ = root.descendants().reverse().filter( d => d.data.id === ll.target );

        if(src.length > 1){
            window.src = src;
            throw "src.length is too long";
        }

        if(targ.length > 1){
            window.targ = targ;
            throw "targ.length is too long";
        }

        // ::TODO:: Check out what is going on here? There are missing nodes that are returned in the data
        // but not properly shown in the graph viz. There are seeming inconsistencies below.
        //
        if(src.length === 0 && targ.length > 0){
            //missingNodeNames = missingNodeNames.concat(ll.source);
            //console.log("nullSrc: "+ll.source);
            return({type:'nullSrc', source:src[0], target:targ[0]});

        }else if(src.length > 0 && targ.length === 0){
            //var missingNode = {id:ll.target, x:Math.random(), y:Math.random()};
            //console.log("nullTarg: " + ll.target);
            missingNodeNames = missingNodeNames.concat(ll.target);
            return({type: 'nullTarg', source:src[0], target:ll.target});

        }else if(src.length === 0 && targ.length === 0){
            //console.log("nullBoth: " + ll.source+"--"+ll.target);
            //missingNodeNames = missingNodeNames.concat(ll.source);
            //missingNodeNames = missingNodeNames.concat(ll.target); 
            return({type: 'nullBoth', source:null, target:null});

        }else{

            return({type: 'ok', source:src[0], target:targ[0]});

        }
    });
    return({rpl:reqProvLinks, missingNodeNames:missingNodeNames});
}


/**
 * Returns true if the <select> has at least one option selected.
 *
 * @param {HTMLSelectElement|string} select – the element itself or its id
 * @returns {boolean}
 */
function hasSelection(select) {
  const el = typeof select === 'string' ? document.getElementById(select) : select;
  // In a multiple‑select, `selectedOptions` is a live HTMLCollection.
  return el && el.selectedOptions.length > 0;
}

/**
 * Returns an array containing the values (or texts) of all selected options.
 *
 * @param {HTMLSelectElement|string} select – the element itself or its id
 * @param {Object} [options] – optional tweaks
 * @param {'value'|'text'} [options.kind='value'] – whether to return the option's value or its visible text
 * @returns {Array<string>}
 */
function getSelectedList(select, { kind = 'value' } = {}) {
  const el = typeof select === 'string' ? document.getElementById(select) : select;
  if (!el) return [];

  // Convert the HTMLCollection to a real array and map to the desired property.
  return Array.from(el.selectedOptions).map(opt => kind === 'text' ? opt.text : opt.value);
}




export { 
    setNodeSelection,
    closeNodeData,
    clearPlots,
    fetchCalibrationInfoForNode,
    makeMSPlots,
    fetchNodeData,
    fetchTechData,
    fetchEmissionsData,
    fetchRequestedQuantitiesData,
    makeReqProvLinks,
    hasSelection,
    getSelectedList
}
