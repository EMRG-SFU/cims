
// This function should visit every node in the tree depth first, regardless of folded status, and call the function
// on each visited node.
function iterWholeTree(root, ff, ff_leaf){

    function iterHelper(nn){
        if(nn._children !== undefined){
            // Call the function on this nodes children (the _children thing points at all the children all the time, disregarding
            // whether the structure has been folded up or not.
            nn._children.forEach(d => iterHelper(d));

            // Now apply the function `ff` to this node, now that we've done all the children.
            return ff(nn);
        }else{
            // if `_children` is not defined then we are at a leaf. Apply the `ff_leaf` function here
            return ff_leaf(nn);
        }
    }

    iterHelper(root);

}


function foldAll(){

    function ff(nn){
        // Set our child array to null, this accomplishes the fold
        nn.children = null;
        // Go to each of our children and set the folded attribute to `true`.
        nn._children.forEach(d => {d.folded = true;});
    }
    function ff_leaf(nn){
        // No folding activity to do already at a leaf node.
    }

    iterWholeTree(root, ff, ff_leaf);
    update(null, root);
}


function unfoldAll(){

    function ff(nn){
        // Set our child array to null, this accomplishes the fold
        nn.children = nn._children;
        // Go to each of our children and set the folded attribute to `true`.
        nn._children.forEach(d => {d.folded = false;});
    }
    function ff_leaf(nn){
        // No folding activity to do already at a leaf node.
    }

    iterWholeTree(root, ff, ff_leaf);
    update(null, root);
}

export { iterWholeTree, foldAll, unfoldAll }
