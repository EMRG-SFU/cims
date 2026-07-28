
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
        Here we take the cross product of the things in the nodeList, the techList, and the yearList, and for each of
        these (node,tech,year) triples we try to get the `paramName` out of the graph. We return this as a list of
        (node,tech,year,value) dicts, and we also store this here as `self.crossProducts`.
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
        """
        Here we take `self.crossProducts`, filter out all the dicts where the param was `None` (a lot of the cross products
        don't actually exist), and re-store the list of dicts, now just with things that DO exist, as `self.paramDicts`.
        """
        self.paramDicts = [a for a in self.crossProducts if a[self.paramName] is not None]


    def setNodes(self, node):

        nodeList = []

        # If node is already a list of thing we need to iterate over them.
        if isinstance(node, Iterable) and not isinstance(node, str):  # Extra case as I think `str`s are also iterable.
            for nn in node:
                if isinstance(nn, str):
                    # Put the node into the nodeList
                    nodeList.append(nn)

                elif isinstance(nn, RegexSearch):
                    # Generate a list of nodes by searching the regex `nn` against all nodes, and add them to the nodelist
                    nodeMatches = [a for a in self.initNodeSet if re.search(nn.pattern, a, flags=re.IGNORECASE)]
                    nodeList.extend(nodeMatches)

                elif isinstance(nn, RegexMatch):
                    nodeMatches = [a for a in self.initNodeSet if re.match(nn.pattern, a, flags=re.IGNORECASE)]
                    nodeList.extend(nodeMatches)

                else:
                    raise RuntimeError("Node spec. must be a string or a Regex object.")

        else:
            # Otherwise it's a single thing and not a list, so process it, similar to the above.
            if isinstance(nn, str):
                # Put the node into the nodeList
                nodeList.append(nn)

            elif isinstance(nn, RegexSearch):
                # Generate a list of nodes by searching the regex `nn` against all nodes, and add them to the nodelist
                nodeMatches = [a for a in self.initNodeSet if re.search(nn.pattern, a, flags=re.IGNORECASE)]
                nodeList.extend(nodeMatches)

            elif isinstance(nn, RegexMatch):
                nodeMatches = [a for a in self.initNodeSet if re.match(nn.pattern, a, flags=re.IGNORECASE)]
                nodeList.extend(nodeMatches)

            else:
                raise RuntimeError("Node spec. must be a string or a Regex object.")

        self.nodeList = nodeList
        self.possibleTechs = UF.getAllTechNames(self.graph, self.nodeList)


    def setTechs(self, tech):

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


