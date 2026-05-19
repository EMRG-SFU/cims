
import marimo as mo
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
import sys
import os, os.path
import re
from collections.abc import Iterable
import types

import Calibration.utility_functions as UF
import Calibration.jupyterOpt as JO
import Calibration.optimization as OO
import Calibration.optimization_seqYears as OptSeq
from Calibration.optimization_seqYears import dbgStruct
import Calibration.plotting as plotting



def find_calibration_nodes(g, searchStr = r'calibration', retAll=False):
   
    ret = UF.findTechsWithParam_anyYears(g, searchStr)
    if retAll:
        return(ret)
    else:
        return(sorted([b for b in set([a['node'] for a in ret])]))


class Calibrator:


    def __init__(self, model, initialNodeSet):
        self.model = model
        self.initialNodeSet = initialNodeSet
        


    def _getFICs(self, loc):

        if isinstance(loc, str):
            # Here we assume that loc is the string name of a node, we get all the techs and years based on that
            # and we display them
            allYears = UF.getAllNodeYears(self.model.graph, loc, asStr=True)
            allTechNames = UF.getAllTechNames(self.model.graph, loc)
            retDict = {"techName":allTechNames}
            retDict.update(
                {f"y_{yv}":[self.model.get_param('fic', loc, year=yv, tech=tv) for tv in allTechNames] for yv in allYears}
            )
            return(retDict)

        elif isinstance(loc, ParamLoc):
            # Here the ParamLoc object contains triples of node, tech, and year. We display all of these in a dataframe.
            # If we can assume that paramDicts are logically constructed dicts containing the node, tech, year, and param value, then
            # we should be able to turn this directly into a dataframe, and pivot the years into the columns.
            return([UF.omit_keys(d, keys_to_remove=['error']) for d in loc.paramDicts])

    def _setFICs(self, node, data):
        pass   


    def inspectFICs(self, loc):

        #if loc is None:
        #    loc = self.paramLoc
        
        if isinstance(loc, str):
            retDict = self._getFICs(loc)
            return( 
                   pl.DataFrame(retDict)
            )
        elif isinstance(loc, ParamLoc):
            loc.prepare(self)
            retDict = self._getFICs(loc)
            return(
                pl.DataFrame(retDict).pivot(on="year", values=loc.paramName)
            )

    def printFICs(self, loc=None):
        """
        Some FIC printing routines that can produce summary statistics over defined sets of nodes.
        """
        pass
        

    def plotFICs(self, node):
        """
        Some FIC plotting routines that can produce summary statistics over defined sets of nodes.
        """
        pass

###############################
###############################

    def tweakFICs(self, loc):

        #if loc is None:
        #    loc = self.paramLoc

        if isinstance(loc, str):

            retDict = self._getFICs(loc)

            # ::TODO:: Here get the form submit function to set the fic value in the model/graph.

            return( 
                   mo.ui.data_editor(pl.DataFrame(retDict)).form(on_change=lambda value: print(f"Submitted thingy: {value}"))
            )
        elif isinstance(loc, ParamLoc):
            loc.prepare(self)
            retDict = self._getFICs(loc)
            return(
                mo.ui.data_editor(pl.DataFrame(retDict).pivot(on="year", values=loc.paramName)).form(on_change=lambda value: print(f"Submitting {value}"))
            )
                                  

    def setFICs(self, loc, value):
        """
        `paramLocs` is a list of paramLoc objects. Each of these
        contain a list of nodes, a list of techs, and a list of years. The list can have a single item, or it can also
        be a regular expression. The ParamLoc specifies spots where the FIC is to be set by calculating the cross
        product of each of these three inner lists.

        `value` is the FIC value to be set. This can be a single value, or it can be a function, in which case the
        function is executed given the node, the tech, the year, and the current FIC value. This is so that you can say
        "ok for everything under dwellings, for all 'electricity' techs, if the current FIC is between 0.0 and 20.0 then
        set it to 10.0, but if it's greater than that, double it. But only for years greater than 2005.
        """

        loc.prepare(self)

        retDict = self._getFICs(loc)

        if isinstance(value, types.FunctionType):
            for item in retDict:
                newValue = value(node=item['node'], tech=item['technology'], year=item['year'], oldValue=item['fic'])
                self.model.set_param_calibration(newValue, 'fic', item['node'], year=item['year'], tech=item['technology'], save=False)
        else:
            for item in retDict:
                self.model.set_param_calibration(value, 'fic', item['node'], year=item['year'], tech=item['technology'], save=False)


###############################
###############################

    def plotMS(self, loc):

        if isinstance(loc, str):

            plotting.plot_ms_for_node(self.model, loc)

        elif isinstance(loc, ParamLoc):

            raise NotImplemented("Can't call plotMS with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")


    def optimize_node(self, loc):

        if isinstance(loc, str):

            OO.optimize(self.model, loc, init_x="zero")

        elif isinstance(loc, ParamLoc):

            raise NotImplemented("Can't call optimize with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")


    def optimize_node_oneYear(self, loc, year):

        if isinstance(loc, str):
            OO.optimize_one_year(self.model, loc, year, init_x="zero")

        elif isinstance(loc, ParamLoc):
            raise NotImplemented("Can't call optimize_oneYear with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def optimize_seqYears(self, loc, logFile=None):
        """
        This one solves a separate optimization problem for each year sequentially. I.e. the year 2005 is
        fully optimized (in terms of tweaking 2005's FICs to minimize the difference between the predicted
        and the actual total market share, for 2005) before the year 2010 is optimized, and after 2000 is.
        The assumption here is that the best possible fit (or at least a fit that works and is good enough) can
        be broken down like this. This technique could not deliberately select a worse fit in an earlier year in 
        order that a much better fit could be acheived in a later year. Whether that sort of thing ever occurs is
        a valid question... and if it could it would perhaps signify that something about the whole process is 
        inappropriate.
        """
        if isinstance(loc, str):
            if logFile is None:
                OptSeq.optimize(self.model, loc, init_x="zero")
            else:
                OptSeq.optimize(self.model, loc, init_x="zero", logFile=logFile)

        elif isinstance(loc, ParamLoc):
            raise NotImplemented("Can't call OptSeq.optimize with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def optimize_seqYears_calibrationHack(self, loc, logFile=None):
        """
        This is identical to `optimize_seqYears`, but it uses experimental "shortened/optimized" versions of
        the LCC calculation function and perhaps also the stock allocation function. Some of these do more computation
        than is strictly required for generating total market shares within a single service node, so we experiment with 
        chopping some of this out to see if performance improves while still computing the same values.
        """
        if isinstance(loc, str):
            if logFile is None:
                OptSeq.optimize_calibrationHack(self.model, loc, init_x="zero")
            else:
                OptSeq.optimize_calibrationHack(self.model, loc, init_x="zero", logFile=logFile)

        elif isinstance(loc, ParamLoc):
            raise NotImplemented("Can't call OptSeq.optimize_calibrationHack with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def optimize_seqYears_calibrationHack_bayes(self, loc, logFile=None):
        """
        This is identical to `optimize_seqYears`, but it uses experimental "shortened/optimized" versions of
        the LCC calculation function and perhaps also the stock allocation function. Some of these do more computation
        than is strictly required for generating total market shares within a single service node, so we experiment with 
        chopping some of this out to see if performance improves while still computing the same values.
        """
        if isinstance(loc, str):
            if logFile is None:
                OptSeq.optimize_calibrationHack_bayes(self.model, loc)
            else:
                OptSeq.optimize_calibrationHack_bayes(self.model, loc, logFile=logFile)

        elif isinstance(loc, ParamLoc):
            raise NotImplemented("Can't call OptSeq.optimize_calibrationHack_bayes with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def optimize_seqYears_basinHop(self, loc, logFile=None):
        """
        This one does pretty much the same thing as `optimize_seqYears` above, but runs a basinhopping
        meta-optimization OF the L-BFGS-B minimization at each year
        """
        if isinstance(loc, str):
            if logFile is None:
                OptSeq.optimize_basinHop(self.model, loc, init_x="zero")
            else:
                OptSeq.optimize_basinHop(self.model, loc, init_x="zero", logFile=logFile)

        elif isinstance(loc, ParamLoc):
            raise NotImplemented("Can't call OptSeq.optimize with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def recompute_node(self, loc):

        if isinstance(loc, str):

            JO.rerunNode(self.model, loc)

        elif isinstance(loc, ParamLoc):

            raise NotImplemented("Can't call optimize with a ParamLoc just yet. Not implemented.")

        else:
            raise RuntimeError("loc must be a Node name (str) or a ParamLoc object")

    def writeCSV(self, loc, filename):

        if isinstance(loc, str):
            retDict = self._getFICs(loc)

            # ::TODO:: Here get the form submit function to set the fic value in the model/graph.
            pl.DataFrame(retDict).write_csv(file=filename)
            print(f"CSV written to: {filename}")
            return(True)

        elif isinstance(loc, ParamLoc):
            loc.prepare(self)
            retDict = self._getFICs(loc)
            pl.DataFrame(retDict).pivot(on="year", values=loc.paramName).write_csv(file=filename)
            print(f"CSV written to: {filename}")
            return(True)

###############################
###############################

    def findNodes(self, searchRE = "calibration"):
        return(
            UF.findTechsWithParam_anyYears(self.model.graph, searchRE, returnDict = True) 
        )

class RegexSearch:
    def __init__(self, string):
        self.pattern = string

class RegexMatch:
    def __init__(self, string):
        self.pattern = string

class All:
    def __init__(self):
        pass


class TechNotFound(Exception):
    pass


class ParamLoc:

    def __init__(self, nodes, techs, years, paramName='fic'):

        self.inputNodes = nodes
        self.inputTechs = techs
        self.inputYears = years
        self.inputParamName = paramName

    def prepare(self, calibrator):
        self.calibrator = calibrator
        self.model = calibrator.model
        self.graph = calibrator.model.graph
        self.allNodes = list(calibrator.model.graph.nodes())
        self.initNodeSet = calibrator.initialNodeSet

        self.setNodes(self.inputNodes)
        self.setTechs(self.inputTechs)
        self.setYears(self.inputYears)
        self.getCrossProductParams(paramName=self.inputParamName)
        self.consolidate()


    def getCrossProductParams(self, paramName='fic'):
        """
        Here we take the cross product of the things in the nodeList, techList, and yearList, and for each of those
        triples we try to get the `paramName` out of the graph.
        """
        outputDicts = []
        for nn in self.nodeList:
            for tt in self.techList:
                for yy in self.yearList:
                    try:
                        p = self.model.get_param(paramName, nn, year=yy, tech=tt)
                        err = None
                        outputDicts.append( {'node':nn, 'technology':tt, 'year':yy, paramName:p, 'error':err} )
                    except Exception as e:
                        p = None
                        err = e
                        outputDicts.append( {'node':nn, 'technology':tt, 'year':yy, paramName:p, 'error':err} )
        self.paramName = paramName
        self.crossProducts = outputDicts
        return(outputDicts)

    def consolidate(self):
        self.paramDicts = [a for a in self.crossProducts if a[self.paramName] is not None]


    def setNodes(self, node):

        # Do the nodes
        nodeList = []
        # If node is already a list of things we need to iterate over them.
        if isinstance(node, Iterable) and not isinstance(node, str):

            for nn in node:

                if isinstance(nn, str):
                    # Put the node into the nodeList
                    nodeList.append(nn)

                elif isinstance(nn, RegexSearch):
                    # Generate a list of nodes by matching the Regex against all nodes, and add
                    # these to the nodeList.
                    nodeMatches = [a for a in self.initNodeSet if re.search(nn.pattern, a, flags=re.IGNORECASE)]
                    nodeList.extend(nodeMatches)

                elif isinstance(nn, RegexMatch):
                    # Generate a list of nodes by matching the Regex against all nodes, and add
                    # these to the nodeList.
                    nodeMatches = [a for a in self.initNodeSet if re.match(nn.pattern, a, flags=re.IGNORECASE)]
                    nodeList.extend(nodeMatches)

                else:
                    raise RuntimeError("Node spec. must be a string or a Regex object")

        # Otherwise it's a single node, or a regex, so process these.
        else:

            if isinstance(node, str):
                nodeList.append(node)

            elif isinstance(node, RegexSearch):
                nodeMatches = [a for a in self.initNodeSet if re.search(node.pattern, a, flags=re.IGNORECASE)]
                nodeList.extend(nodeMatches)

            elif isinstance(node, RegexMatch):
                nodeMatches = [a for a in self.initNodeSet if re.match(node.pattern, a, flags=re.IGNORECASE)]
                nodeList.extend(nodeMatches)

            else:
                raise RuntimeError("Node spec. must be a string or a Regex object")

        self.nodeList = nodeList
        self.possibleTechs = UF.getAllTechNames_raiseIfInconsistent(self.graph, self.nodeList)

    def setTechs(self, tech):
        # Do the techs now, using the `self.possibleTechs` to do the filtering
        
        techList = []

        if isinstance(tech, Iterable) and not isinstance(tech, str):

            for tt in tech:

                if isinstance(tt, str):
                    if tt in self.possibleTechs:
                        techList.append(tt)
                    else:
                        raise TechNotFound(f"Tech: {tt} is not found in any of the selected node set's techs.")

                elif isinstance(tt, RegexSearch):
                    techMatches = [a for a in self.possibleTechs if re.search(tt.pattern, a, flags=re.IGNORECASE)]
                    if len(techMatches) > 0:
                        techList.extend(techMatches)
                    else:
                        raise TechNotFound(f"Tech regex pattern: {tt.pattern} doesn't match any possible techs in selected node set.")

                elif isinstance(tt, RegexMatch):
                    techMatches = [a for a in self.possibleTechs if re.match(tt.pattern, a, flags=re.IGNORECASE)]
                    if len(techMatches) > 0:
                        techList.extend(techMatches)
                    else:
                        raise TechNotFound(f"Tech regex pattern: {tt.pattern} doesn't match any possible techs in selected node set.")

                else:
                    raise RuntimeError("Tech spec. must be a string or a Regex object")

        else:

            if isinstance(tech, str):
                if tech in self.possibleTechs:
                    techList.append(tech)
                else:
                    raise TechNotFound(f"Tech: {tech} not found in any of the selected node set's techs.")

            elif isinstance(tech, RegexSearch):
                techMatches = [a for a in self.possibleTechs if re.search(tech.pattern, a, flags=re.IGNORECASE)]
                if len(techMatches) > 0:
                    techList.extend(techMatches)
                else:
                    raise TechNotFound(f"Tech regex pattern: {tech.pattern} doesn't match any possible techs in selected node set.")

            elif isinstance(tech, RegexMatch):
                techMatches = [a for a in self.possibleTechs if re.match(tech.pattern, a, flags=re.IGNORECASE)]
                if len(techMatches) > 0:
                    techList.extend(techMatches)
                else:
                    raise TechNotFound(f"Tech regex pattern: {tech.pattern} doesn't match any possible techs in selected node set.")

            elif isinstance(tech, All):
                techMatches = self.possibleTechs
                if len(techMatches) > 0:
                    techList.extend(techMatches)
                else:
                    raise TechNotFound("It seems like self.possibleTechs is an empty list? Something weird is going on.")
            else:
                raise RuntimeError("Tech spec. must be a string, a Regex object, or an All object.")

        self.techList = techList

    def setYears(self, year):

        yearList = []

        if isinstance(year, Iterable) and not isinstance(year, str):
            for yy in year:
                if isinstance(yy, str):
                    yearList.append(yy)
                elif isinstance(yy, RegexSearch):
                    raise NotImplemented("RegexSearch for years not yet implemented.")
                elif isinstance(yy, RegexMatch):
                    raise NotImplemented("RegexMatch for years not yet implemented.")
                else:
                    raise RuntimeError("Year must be a string year, or a regexp")

        else:
            if isinstance(year, str):
                yearList.append(year)
            elif isinstance(year, RegexSearch):
                raise NotImplemented("RegexSearch for years not yet implemented.")
            elif isinstance(year, RegexMatch):
                raise NotImplemented("RegexMatch for years not yet implemented.")
            elif isinstance(year, All):
                # Here we return the union of all years found in all the nodes in our selected set so far
                yearList.extend(
                    UF.union_of_sublists([UF.getAllNodeYears(self.graph, a) for a in self.nodeList])
                ) 

        self.yearList = yearList










