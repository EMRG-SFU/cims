"use strict";

import Step from './Step.js'
import { 
    setNodeSelection,
    fetchNodeData, 
    fetchTechData,
    fetchEmissionsData,
    fetchRequestedQuantitiesData,
    fetchCalibrationInfoForNode,
    makeReqProvLinks,
    hasSelection,
    getSelectedList,
    makeMSPlots
} from './CIMSapi.js'

import {
    drawMiniBars,
    drawMiniPie,
    drawSparkline
} from './calNodeGraphics.js'

window.hasSelection = hasSelection;
window.getSelectedList = getSelectedList;

import {
    foldAll,
    unfoldAll,
    iterWholeTree
} from './TreeOps.js'

import {
    populateNodeList,
    populateParamLists,
    filterFunction,
    filterFunction_nodeParams,
    filterFunction_techParams
} from './node_tech_parameters.js'

window.foldAll = foldAll;
window.unfoldAll = unfoldAll;

const width = 1628;
const initHeight = window.innerHeight;
const marginTop = 10;
const marginRight = 10;
const marginBottom = 10;
const marginLeft = 40;

var missing_x_coord = -350;
var missing_y_gap = 50;

var treeInput;
var graphInput;



var missingNodesPre;
var missingNodeData;
var missingNodeDataNames;


var rplObj;

function resizeSelectToFit(sel, maxVH = 90) {
  sel.style.height = 'auto';               // clear any previous explicit height
  const needed = sel.scrollHeight;         // full content height
  console.log("Needed is " + needed);
  const maxPx = (maxVH / 100) * window.innerHeight;
  sel.style.height = Math.min(needed, maxPx) + 'px';
}


// Load all CIMS node names into the selection widget.
populateNodeList();

// Load all the node and tech param names into the selectors.
populateParamLists();

// Functions for dealing with the cims node selection box and its filtering
function toggleDropdown() {
    document.getElementById("myDropdown").classList.toggle("show");
}
window.toggleDropdown = toggleDropdown;

function toggleNodeParams() {
    document.getElementById("myDropdown_nodeParams").classList.toggle("show");
    var sel = document.getElementById("selectNodeParams");
    resizeSelectToFit(sel, 90);
}
window.toggleNodeParams = toggleNodeParams;

function toggleTechParams() {
    document.getElementById("myDropdown_techParams").classList.toggle("show");
    var sel = document.getElementById("selectTechParams");
    resizeSelectToFit(sel, 90);
}
window.toggleTechParams = toggleTechParams;

// This is a little function for handling clicks on the nodeList dropdown that lives near the top.
function spanClicked(x) {
    var selNodeName = x.textContent || x.innerText;
    console.log("Selected node: " + selNodeName);
    loadData(selNodeName);
}
window.spanClicked = spanClicked;


function handleAddressKeydown(event, thing){
    window.kbdEvent = event;
    window.thing = thing;
    if (event.code === 'Enter'){
        var selNodeName = thing.value;
        console.log("Selected node via textfield: " + selNodeName);
        loadData(selNodeName);
    }else{
        // This is just normal textfield typing, don't do anything.
    }
}
window.handleAddressKeydown = handleAddressKeydown


function loadData(rootNodeName) {

    // Write the node address into the textfield
    d3.select("#rootNodeAddress").node().value = rootNodeName;

    let subGraphURL = SERVER_BASE_URL + "getSubgraph/" + rootNodeName
    d3.json(subGraphURL)
    .then(json => {

        treeInput = json['tree']; window.treeInput = treeInput;
        graphInput = json['graph']; window.graphInput = graphInput;
        
        initializeDisplay();

        rplObj = makeReqProvLinks(root); window.rplObj = rplObj;
        var missingNodeSet = new Set(rplObj.missingNodeNames)
        missingNodeDataNames = [...missingNodeSet]; 
        // Set the random x and y fractions for the missing nodes
        missingNodeData = missingNodeDataNames.map((d,i) => {
            var newObj = {id:d, x0:(i+1)*missing_y_gap, y0:missing_x_coord, fixed:true};
            return(newObj);
        });
        console.log(missingNodeDataNames.sort());
        console.log("Do we see: " + SERVER_BASE_URL);
        rplObj.rpl = rplObj.rpl.map(d => {
            if(d.type !== 'nullTarg'){
                return(d);
            }else{
                var newTarg = missingNodeData.filter(n => {
                    //console.log(d.target+ " -- " + n.id);
                    return(d.target === n.id);
                });
                if(newTarg.length !== 1){
                    throw "this this is the wrong length";
                }
                return({type:'nullTarg', source:d.source, target:newTarg[0]});
            }
        });

        window.missingNodeData = missingNodeData;
        
        // arbitrary init code
        root.x0 = dy / 2;
        root.y0 = 0;
        root.descendants().forEach((d,i) => {
            d.id = i;
            d._children = d.children;
            //if (d.depth && d.data.id.length > 7) d.children = null;
        });

        update(null, root);

        //initializeSimulation();
    })
    .catch(error => {
        console.error(error);
    });
}


// Shutting this off for now (because some smaller test graphs being used don't actually have BC's coal mining in them.
// So nothing will be loaded by default, but the selectors should be populated with things that ARE there, so when the
// thing is done loading, head up there and select something from either the populated nodeList, or from the autocompleting 
// text box thing.
//loadData("CIMS.CAN.BC.Coal Mining")

// Turning this back on for manual calibration debugging purposes. Am always having to navigate to this subtree, so
// let's just do it on load.
loadData("CIMS.CAN.AB.Residential");

// Playing now with a bunch of subgraphs, and the above BC residential node isn't in the graph anymore, so this
// fails here and crashes everything.
//loadData("CIMS.CAN.AB")

// This function is pretty much identical to the one above, but can be run at the console. Supply the filename
// that should be loaded (where the file is a json file structured as before/above).
function loadData_manualFile(fname){
    d3.json(fname)
        .then(json => {

        treeInput = json['tree']; window.treeInput = treeInput;
        graphInput = json['graph']; window.graphInput = graphInput;
        
        initializeDisplay();

        rplObj = makeReqProvLinks(root); window.rplObj = rplObj;
        var missingNodeSet = new Set(rplObj.missingNodeNames)
        missingNodeDataNames = [...missingNodeSet]; 
        // Set the random x and y fractions for the missing nodes
        missingNodeData = missingNodeDataNames.map((d,i) => {
            var newObj = {id:d, x0:(i+1)*missing_y_gap, y0:missing_x_coord, fixed:true};
            return(newObj);
        });
        rplObj.rpl = rplObj.rpl.map(d => {
            if(d.type !== 'nullTarg'){
                return(d);
            }else{
                var newTarg = missingNodeData.filter(n => d.target === n.id);
                if(newTarg.length !== 1){
                    throw "this this is the wrong length";
                }
                return({type:'nullTarg', source:d.source, target:newTarg[0]});
            }
        });

        window.missingNodeData = missingNodeData;
        
        // arbitrary init code
        root.x0 = dy / 2;
        root.y0 = 0;
        root.descendants().forEach((d,i) => {
            d.id = i;
            d._children = d.children;
            //if (d.depth && d.data.id.length > 7) d.children = null;
        });

        update(null, root);

        //initializeSimulation();
    })
    .catch(error => {
        console.error(error);
    });

}
window.loadData_manualFile = loadData_manualFile;




function customStep(context) {
  return new Step(context, 0.75);
}

var svg = d3.select("svg");

var root;
var dx = null;
var dy = null;
var tree = null;
var diagonal = null;
var techLink = null;
var gLink = null;
var rpLinkPre = null;
var rpLinkPre_nullTarg = null;
var gNode = null;
var theMarker = null;
var theMarker_rp = null;
var theMarker_missing = null;

var zoom = null;

function initializeDisplay() {

    // Clear off the svg
    svg.selectAll("*").remove();


    root = d3.hierarchy(treeInput);
    window.root = root;
    // This (`dx`) influences how much vertical separation there is between nodes in the tree
    dx = 40; //23;
    // This (`dy`) influences how much horizontal separation there is between a parent node and its children.
    //dy = (width - marginRight - marginLeft) / (1 + root.height) + 30;
    dy = (width - marginRight - marginLeft) / (1 + root.height) + 50;
    
    
    tree = d3.tree()
        .nodeSize([dx,dy])
        .separation(function separation(a,b) {
            return a.parent == b.parent ? 1 : 2.5;
        });
    diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);
    techLink = d3.link(customStep).x(d => d.y).y(d => d.x);

    svg
        .attr("width", width)
        .attr("height", initHeight)
        //.attr("viewBox", [-marginLeft, -marginTop, width, dx])
        // This makes wonky.
        //.attr("preserveAspectRatio", "none")
        //.attr("style", "max-width: 100%; height: auto; font: 20px sans-serif; user-select: none;");
        .attr("style", "max-width: 100%; font: 20px sans-serif; user-select: none;")
        .on("click", (event,d) => {
            //console.log("Firing SVG click event");
            //window.wtfEvent = event;
            //window.wtfD = d;
            console.log("Position: " + wtfTransform.invert(d3.pointer(event)));
        });

    gLink = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", 1.5);

    rpLinkPre = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "purple")
        .attr("stroke-opacity", 0.5)
        .attr("stroke-width", 5.0);

    rpLinkPre_nullTarg = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-opacity", 0.5)
        .attr("stroke-width", 3.0);

    gNode = svg.append("g")
        .attr("id", "nodesContainer")
        .attr("cursor", "pointer")
        .attr("pointer-events", "all");

    missingNodesPre = svg.append("g")
        .attr("cursor", "pointer")
        .attr("pointer-events", "all");

    function zoomed({transform}){
        gLink.attr("transform", transform);
        gNode.attr("transform", transform);
        rpLinkPre.attr("transform", transform);
        rpLinkPre_nullTarg.attr("transform", transform);
        missingNodesPre.attr("transform", transform);

        window.wtfTransform = transform;
        
    }

    zoom = d3.zoom()
        .extent([[0,0], [width, dx]])
        .on("zoom", zoomed);

    svg.call(zoom);
    window.zoom = zoom;

    theMarker = svg.append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox","0 0 10 10")
        .attr("refX","10")
        .attr("refY","5")
        .attr("markerUnits", "strokeWidth")
        .attr("markerWidth", "10")
        .attr("markerHeight", "5")
        .attr("orient", "auto")
    .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("stroke", "black")
        .attr("fill", "black");

     theMarker_rp = svg.append("marker")
        .attr("id", "arrowhead_rp")
        .attr("viewBox","0 0 10 10")
        .attr("refX","10")
        .attr("refY","5")
        .attr("markerUnits", "strokeWidth")
        .attr("markerWidth", "10")
        .attr("markerHeight", "5")
        .attr("orient", "auto")
    .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("stroke", "purple")
        .attr("fill", "purple");

     theMarker_missing = svg.append("marker")
        .attr("id", "arrowhead_missing")
        .attr("viewBox","0 0 10 10")
        .attr("refX","10")
        .attr("refY","5")
        .attr("markerUnits", "strokeWidth")
        .attr("markerWidth", "10")
        .attr("markerHeight", "5")
        .attr("orient", "auto")
    .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("stroke", "red")
        .attr("fill", "red");

    resetZoom();
}

function resetZoom(){
    svg.call(zoom.transform, d3.zoomIdentity);
}
window.resetZoom = resetZoom;






function update(event, source) {
    
    const duration = event?.altKey ? 2500 : 250;
    const nodes = root.descendants().reverse(); //window.nodes = nodes;
    const links = root.links(); //window.links = links;

    tree(root);

    let left = root;
    let right = root;
    root.eachBefore(node => {
        if (node.x < left.x) left = node;
        if (node.x > right.x) right = node;
    });

    const height = right.x - left.x + marginTop + marginBottom;
    window.height = height;
    window.right = right;
    window.left = left;

    rplObj.rpl = rplObj.rpl.map(d => {
        if(d.type === 'nullTarg'){
            if(d.target.fixed !== true){
                var dNew = d;
                dNew.target.x = dNew.target.x0 * height;
                dNew.target.y = dNew.target.y0 * width;
                return(dNew);
            }else{
                var dNew = d;
                dNew.target.x = dNew.target.x0;
                dNew.target.y = dNew.target.y0; 
                return(dNew);
            }

        }else{
            return(d);
        }
    });

    const transition = svg.transition()
        .duration(duration)
        .attr("height", initHeight)
        //.attr("viewBox", [-marginLeft, left.x - marginTop, width, height])
        // This makes wonky
        //.attr("preserveAspectRatio", "none")
        .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));

    const node = gNode.selectAll("g")
        .data(nodes, d => d.id);

    const nodeEnter = node.enter().append("g")
        .attr("transform", d => `translate(${source.y0},${source.x0})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0)
        .on("click", (event, d) => {

            if(event.shiftKey === true){

                if(event.metaKey === true){
                    // If we've ctrl-shift-clicked on a SERVICE node, we want to do a calibration on it, involving all its techs.
                    // That's what has to happen in here...
                    console.log("Meta-Shift-Clicked on Node/Tech (d.data.id): " + d.data.id);
                    if(d.data.isTechNode){
                        // Don't do anything here for now, as there's no calibration procedure (currently) that involves only a single
                        // technology.
                        console.log("Skipping, as we don't calibrate on a single tech node");
                    }else{
                        // Here's where we do the full calibration thing.
                        fetchCalibrationInfoForNode(d.data.id);
                    }


                }else{
                    // If ctrl isn't down as well, we do what we did before which is just to display all the information that's
                    // available at that particular point.
                    console.log("Shift-Clicked on Node/Tech (d.data.id): " + d.data.id);
                    if(d.data.isTechNode){

                        console.log("Tech data JSON: " + JSON.stringify(d.data));
                        fetchTechData(d.data.id, d.data.techName);

                    }else{

                        // Reset the "is selected" attribute in all nodes.
                        //const nodes = nodesPre.forEach( node => { node.data.is_selected = false; });
                        nodes.forEach( node => { node.data.is_selected = false; });
                        setNodeSelection(d.data.id);
                        d.data.is_selected = true;
                        update(event,d);
                        //fetchNodeData(d.data.id);
                        //fetchEmissionsData(d.data.id);
                        //fetchRequestedQuantitiesData(d.data.id);

                    }
                }

            }else{

                if(d.children === null){

                    d.children = d._children;
                    d.each( d => {
                        d.folded = false;
                    })


                }else{

                    d.each( d => {
                        d.folded = true;
                    })
                    d.children = null;

                }

                // This ternary thing is now expressed in the if/else above
                //d.children = d.children ? null : d._children;
                update(event, d);
            }

        });

    // ::TODO:: I think this is where the little bar graphs or pie charts should be shown.
    nodeEnter.append("circle")
        .attr("r", d => d.data.isTechNode ? 3.0 : 9.0)
        .attr("fill", d => {
            if(d.data.is_selected){
                return("#900");
            }else{
                return( d._children ? "#555" : "#999" );
            }
        })
        .attr("stroke-width", 10);

    nodeEnter.append("text")
        .attr("dy", "0.31em")
        //.attr("x", d => d._children ? -6 : 6)
        .attr("x", d => 24)
        //.attr("text-anchor", d => d._children ? "end" : "start")
        .attr("text-anchor", d => "start")
        .text(d => d.data.id.split(".").reverse()[0])
        //.text(d => d.data.id)
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 10)
        .attr("stroke", "white")
        .attr("fill", d => {
            if(d.data.is_selected){
                console.log("setting red"); 
                return("red");
            }else{
                console.log("setting not red"); 
                return(d.data.isTechNode ? "blue" : "black")
            }
        })
        .attr("paint-order", "stroke");

    const nodeUpdate = node.merge(nodeEnter).transition(transition)
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .attr("fill-opacity", 1)
        .attr("stroke-opacity", 1);

    nodeUpdate.select("text")
        .attr("fill", d => {
            if(d.data.is_selected){
                console.log("setting red"); 
                return("red");
            }else{
                console.log("setting not red"); 
                return(d.data.isTechNode ? "blue" : "black")
            }
        });

    nodeUpdate.select("circle")
        .attr("fill", d => {
            if(d.data.is_selected){
                return("#900");
            }else{
                return( d._children ? "#555" : "#999" );
            }
        });

    const nodeExit = node.exit().transition(transition).remove()
        .attr("transform", d => `translate(${source.y},${source.x})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0);


    const missingNodes = missingNodesPre.selectAll("g")
        .data(missingNodeData);

    const missingNodesEnter = missingNodes.enter().append("g")
        .attr("transform", d => `translate(0,0)`)
        .attr("class", "missingNode")
        .attr("fill-opacity", 1)
        .attr("stroke-opacity", 1);

    //missingNodesEnter.append("circle")
    //    .attr("r", 3,5)
    //    .attr("fill", "green")
    //    .attr("stroke-width", 5);

    missingNodesEnter.append("text")
        .attr("dy", "0.0em")
        .attr("x", d => 0)
        .attr("text-anchor", d => "start")
        .text(d => d.id)
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 7)
        .attr("stroke", "white")
        .attr("stroke-opacity", 0.7)
        .attr("fill", "red")
        .attr("paint-order", "stroke");
    
    const missingNodesUpdate = missingNodes.merge(missingNodesEnter).transition(transition)
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .attr("fill-opacity", 1)
        .attr("stroke-opacity", 1);

    const missingNodesExit = missingNodes.exit().transition(transition).remove()
        .attr("transform", `translate(0,0)`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0);

    window.missNodes = missingNodes;

    const link = gLink.selectAll("path")
        .data(links, d => d.target.id);

    const linkEnter = link.enter().append("path")
        .attr("marker-end", d => {
            if(d.target.data.isTechNode){
                return(null);
            }else if(d.target.data.isReqProv){
                //return("url(#arrowhead_rp)");
                return(null);
            }else{
                //return("url(#arrowhead)");
                return(null);
            }
        })
        .attr("d", d => {
            const o = {x: source.x0, y: source.y0};
            if(d.target.data.isTechNode){
                return techLink({source: o, target: o});
            }else{
                return diagonal({source: o, target: o});
            }
        })
        .attr("stroke", d => d.target.data.isTechNode ? "orange" : "grey")
        .attr("stroke-opacity", d => d.target.data.isTechNode ? 1.0 : 0.4);

    link.merge(linkEnter).transition(transition)
        .attr("d", d => {
            //const o = {x: source.x0, y: source.y0};
            if(d.target.data.isTechNode){
                //return techLink({source: o, target: o});
                //window.wtfTech = d;
                return techLink(d);
            }else{
                //return diagonal({source: o, target: o});
                //window.wtfBlah = d;
                return diagonal(d);
            }
        })
        .attr("stroke", d => { 
            if(d.target.data.isTechNode){
                return("orange");
            }else if(d.target.data.isReqProv){
                return("purple");
            }else{
                return("black");
            }
        })
        .attr("stroke-width", d => { 
            if(d.target.data.isTechNode){
                return(2.0);
            }else if(d.target.data.isReqProv){
                return(5.0);
            }else{
                return(1.5);
            }
        })
        .attr("stroke-dasharray", d => { 
            if(d.target.data.isTechNode){
                return(null);
            }else if(d.target.data.isReqProv){
                return(null);
            }else{
                return("15 7");
            }
        })
        .attr("stroke-opacity", d => { 
            if(d.target.data.isTechNode || d.target.data.isReqProv){
                return(0.6);
            }else{
                return(1.0);
            }
        });

    link.exit().transition(transition).remove()
        .attr("d", d => {
            const o = {x: source.x, y: source.y};
            if(d.target.data.isTechNode){
                return techLink({source: o, target: o});
            }else{
                return diagonal({source: o, target: o});
            }
        });

    /////////// Now the graph links

    function checkNodesFolded(d){
        let s = (d.source.folded === undefined) || (d.source.folded === false);
        let t = (d.target.folded === undefined) || (d.target.folded === false);
        if(s && t){
            return true;
        }else{
            return false;
        }
    }
    // This actually works for making some kind of custom curve thing.
    // Note the very procedural curve construction in the anonymous function
    // writing the 'd' attribute.
    const rpLink = rpLinkPre.selectAll("path")
        .data(rplObj.rpl.filter(d => (d.type === 'ok') && (checkNodesFolded(d))))
        //.data([])
        .join("path")
        .attr("stroke", "purple")
        .attr("opacity", d => { 
            let showMissing = document.getElementById("reqProvLinkCheck").checked;
            if(showMissing){
                return(1.0);
            }else{
                return(0.0);
            }
        })
        .attr("marker-end", "url(#arrowhead_rp)")
        .attr("d", d => {
            const path = d3.path();
            const curve = d3.curveBundle(path);
            curve.lineStart();
            curve.point(d.source.y, d.source.x);
            curve.point(width*2,d.source.x);
            curve.point(d.target.y, d.target.x);
            curve.lineEnd();
            return(path);
        });
    //    .attr("x1", d => d.source.y)
    //    .attr("y1", d => d.source.x)
    //    .attr("x2", d => d.target.y)
    //    .attr("y2", d => d.target.x);

    const rpLink_nullTarg = rpLinkPre_nullTarg.selectAll("path")
        .data(rplObj.rpl.filter(d => (d.type === 'nullTarg') && (checkNodesFolded(d))))
        .join("path")
        .attr("stroke", "red")
        .attr("opacity", d => { 
            let showMissing = document.getElementById("missingLinkCheck").checked;
            if(showMissing){
                return(1.0);
            }else{
                return(0.0);
            }
        })
        .attr("marker-end", "url(#arrowhead_missing)")
        .attr("d", d => {
            const path = d3.path();
            const curve = d3.curveBundle(path);
            curve.lineStart();
            curve.point(d.source.y, d.source.x);
            curve.point(-width,d.source.x);
            //curve.point(width*1.5, 0.0);
            curve.point(d.target.y, d.target.x);
            curve.lineEnd();
            return(path);
        });

    root.eachBefore(d => {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}
window.update = update;


