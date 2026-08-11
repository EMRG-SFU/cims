import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
from operator import itemgetter
import types

import Calibration.Data.node_info as node_info

def numFormat(x):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    return f"{x:.2f}"  



def get_emissions_calibration(model, nodeName, key="calibration_emissions_by_type", getDict=False):
    """
    Retrieve calibration emissions and create dataframe with identifying info in columns, and emissions
    values across years
    """
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allEmDict = {yy:nodeDict[yy][key] for yy in yearHeaders}

    calYearEmissions_pre = [{'year':yk, 'gas': gg, 'value': numFormat(yearDict["year_value"])} for yk,emDict in allEmDict.items() for gg,yearDict in emDict.items()]
    if getDict:
        return calYearEmissions_pre

    calYearEmissions = pl.DataFrame(calYearEmissions_pre)
    calYearEmissions_pivot = calYearEmissions.pivot(on="year", values="value")
    return calYearEmissions_pivot

def get_emissions(model, nodeName, key="emissions_total_cumul_net", getDict=False):
    """
    Retrieve emissions object and create dataframe with identifying info in columns, and emissions
    values across the years.
    """
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    # Here we're just interested in the `key` rowName (and this extra structure is the reason that Emissions (and Quantities)
    # always showed up as an error in the 
    allEmDict = {yy:nodeDict[yy][key]["year_value"].emissions for yy in yearHeaders}

    nodeYearEmissions_pre = [{'year':yk, 'fuel':k, 'gas':kk, 'type':kkk, 'value':numFormat(vvv['year_value'])} 
     for yk,emDict in allEmDict.items() 
     for k,v in emDict.items()
     for kk,vv in v.items()
     for kkk,vvv in vv.items()]
    
    if getDict:
        return nodeYearEmissions_pre

    nodeYearEmissions = pl.DataFrame(nodeYearEmissions_pre)
    #print(nodeYearEmissions)
    nodeYearEmissions_pivot = nodeYearEmissions.pivot(on="year", values="value")
    return nodeYearEmissions_pivot


def get_emissions_both_dict(model,
                            nodeName,
                            key_cims = "emissions_total_cumul_net",
                            key_cal = "calibration_emissions_by_type",
                            missingValFunc = None):
    """

    """
     
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    cimsFrame = pl.DataFrame(get_emissions(model, nodeName, getDict=True))
    cimsFrame = cimsFrame.with_columns(pl.col(["value"]).cast(pl.Float64))
    cimsFrame = cimsFrame.group_by(["year","gas"]).agg(pl.col(["value"]).sum()).sort("year")

    calFrame = pl.DataFrame(get_emissions_calibration(model, nodeName, getDict=True))
    calFrame = calFrame.with_columns(pl.col(["value"]).cast(pl.Float64))

    return {'cims': cimsFrame, 
            'cal': calFrame}


def get_emissions_diff_frame(model,
                             nodeName,
                             key_cims = "emissions_total_cumul_net",
                             key_cal = "calibration_emissions_by_type",
                             missingValFunc = None):

    cims, cal = itemgetter('cims','cal')(get_emissions_both_dict(model,
                                                                 nodeName,
                                                                 key_cims,
                                                                 key_cal,
                                                                 missingValFunc))
    cims = cims.rename({'value':'cims_value'})
    cal = cal.rename({'value':'cal_value'})
    both_frame = cims.join(cal, on=['year','gas'], how="inner")
    out_frame = both_frame.with_columns(
            (pl.col('cims_value') - pl.col('cal_value')).alias("diff")
    )
    out_frame = out_frame.with_columns(
            (pl.col('diff') / pl.col('cims_value')).alias("pctDiff_cims"),
            (pl.col('diff') / pl.col('cal_value')).alias("pctDiff_cal")
    )

    return out_frame

############################

# 0x7A69_dark              calmar256-light          developer                hybrid                   mophiaDark               quagmire                 tango-morning
# 256-grayvim              camo                     disciple                 hybrid-light             mophiaSmoke              quiet                    tango2
# 256-jungle               campfire                 distinguished            iangenzo                 mopkai                   radicalgoodspeed         tangoX
# 3dglasses                candy                    django                   ibmedit                  moria                    railscasts               tangoshady
# BlackSea                 candycode                donbass                  icansee                  morning                  rainbow_fine_blue        taqua
# C64                      candyman                 doorhinge                iceberg                  moss                     rainbow_fruit            tchaba
# Chasing_Logic            caramel                  doriath                  impact                   motus                    rainbow_neon             tchaba2
# ChocolateLiquor          carrot                   dual                     impactG                  mrkn256                  random                   tcsoft
# ChocolatePapaya          carvedwood               dull                     industrial               mrpink                   rastafari                telstar
# CodeFactoryv3            carvedwoodcool           dusk                     industry                 mud                      rcg_gui                  tesla
# DevC++                   cascadia                 earendel                 ingretu                  muon                     rcg_term                 tetragrammaton
# Monokai                  catppuccin               earth                    inkpot                   murphy                   rdark                    textmate16
# Monokai-chris            chance-of-storm          earthburn                ir_black                 mustang                  rdark-terminal           thegoodluck
# MountainDew              charged-256              eclipse                  ironman                  native                   redblack                 thestars
# PapayaWhip               charon                   eclm_wombat              jammy                    nature                   redstring                thor
# SlateDark                chela_light              ecostation               jelleybeans              navajo                   refactor                 tibet
# Tomorrow                 chlordane                editplus                 jellybeans               navajo-night             relaxedgreen             tidy
# Tomorrow-Night           chocolate                edo_sea                  jellyx                   nazca                    reliable                 tir_black
# Tomorrow-Night-Blue      chrysoprase              ego                      jhdark                   nedit                    reloaded                 tolerable
# Tomorrow-Night-Bright    ciscoacl                 ekinivim                 jhlight                  nedit2                   retrobox                 tomatosoup
# Tomorrow-Night-Eighties  clarity                  ekvoli                   jiks                     nefertiti                revolutions              tony_light
# abra                     cleanphp                 elda                     kalisi                   neon                     robinhood                toothpik
# adam                     cleanroom                elflord                  kalt                     nerv-ous                 ron                      torte
# adaryn                   clearance                elise                    kaltex                   neutron                  rootwater                transparent
# adobe                    cloudy                   elisex                   kate                     neverland                rtl                      trivial256
# adrian                   clue                     elrodeo                  kellys                   neverland-darker         sand                     trogdor
# advantage                cobalt                   emacs                    khaki                    neverland2               satori                   turbo
# af                       cobaltish                enzyme                   kib_darktango            neverland2-darker        saturn                   tutticolori
# aiseered                 codeblocks_dark          evening                  kib_plastic              neverness                scala                    twilight
# anotherdark              codeburn                 evening_2                kiss                     nevfn                    scite                    twilight256
# ansi_blows               codeschool               far                      kkruby                   newspaper                sea                      twitchy
# apprentice               coffee                   felipec                  koehler                  newsprint                sean                     two2tango
# aqua                     coldgreen                fine_blue                kolor                    nicotine                 seashell                 ubloh
# ashen                    colorer                  flatcolor                kruby                    night                    selenitic                umber-green
# asmanian_blood           colorful                 flatland                 kyle                     nightVision              seoul                    understated
# astronaut                colorful256              flatlandia               landscape                night_vision             seoul256                 underwater
# asu1dark                 colorscheme_template     flattr                   last256                  nightflight              seoul256-light           underwater-mod
# atom                     colorzone                flatui                   lazarus                  nightflight2             settlemyer               unokai
# automation               contrasty                fnaqevan                 legiblelight             nightshimmer             sexy-railscasts          up
# autumn                   cool                     fog                      leglight2                nightsky                 sf                       vanzan_color
# autumnleaf               corn                     fokus                    leo                      nightwish                shadesofamber            vc
# babymate256              corporation              forneus                  less                     no_quarter               shine                    vcbc
# badwolf                  cthulhian                freya                    lettuce                  northland                shobogenzo               vexorian
# base16-atelierdune       custom                   frood                    leya                     northsky                 sienna                   vibrantink
# basic                    d8g_01                   fruidle                  lightcolors              norwaytoday              sift                     vilight
# bayQua                   d8g_02                   fruit                    lilac                    nour                     silent                   visualstudio
# baycomb                  d8g_03                   fruity                   lilydjwg_dark            nuvola                   simple256                vividchalk
# bclear                   d8g_04                   fu                       lilydjwg_green           obsidian                 simple_b                 vj
# beachcomber              dante                    gardener                 lilypink                 obsidian2                simpleandfriendly        void
# beauty256                dark-ruby                gemcolors                lingodirector            oceanblack               simplewhite              vydark
# bensday                  darkBlue                 gentooish                liquidcarbon             oceanblack256            skittles_berry           vylight
# billw                    darkZ                    getafe                   literal_tango            oceandeep                skittles_dark            wargrey
# biogoo                   darkblack                getfresh                 lizard                   oceanlight               slate                    warm_grey
# black_angus              darkblue                 github                   lizard256                olive                    smp                      watermark
# blackbeauty              darkblue2                gobo                     lodestone                orange                   smyck                    whitebox
# blackboard               darkbone                 golded                   louver                   osx_like                 softblue                 whitedust
# blackdust                darkburn                 golden                   lucius                   otaku                    softbluev2               widower
# blacklight               darkdot                  google                   luinnar                  pablo                    softlight                wildcharm
# blazer                   darkeclipse              gor                      luna                     pacific                  sol                      win9xblueback
# blink                    darker-robin             gotham                   lunaperche               paintbox                 sol-term                 winter
# blue                     darkerdesert             gotham256                mac_classic              peachpuff                solarized                wintersday
# bluechia                 darkocean                gothic                   macvim                   peaksea                  sonofobsidian            wombat
# bluedrake                darkrobot                grape                    made_of_code             pencil                   sonoma                   wombat256
# bluegreen                darkroom                 gravity                  mango                    peppers                  sorbet                   wombat256i
# blueprint                darkslategray            graywh                   manuscript               perfect                  sorcerer                 wombat256mod
# blueshift                darkspectrum             grb256                   manxome                  pf_earth                 soso                     wood
# bluez                    darktango                greens                   marklar                  phd                      southernlights           wuye
# blugrine                 darkzen                  greenvision              maroloccio               phphaxor                 southwest-fog            xemacs
# bmichaelsen              darth                    grey2                    mars                     phpx                     spectro                  xian
# bocau                    dawn                     greyblue                 martin_krischik          pic                      spiderhawk               xmaslights
# bog                      default                  grishin                  matrix                   pink                     spring                   xoria256
# borland                  delek                    gruvbox                  mayansmoke               playroom                 stackoverflow            xterm16
# breeze                   delphi                   guardian                 mdark                    pleasant                 stingray                 yaml
# brookstream              denim                    guepardo                 mellow                   potts                    strange                  yeller
# brown                    derefined                h80                      metacosm                 print_bw                 strawimodo               zaibatsu
# bubblegum                desert                   habamax                  midnight                 prmths                   summerfruit              zazen
# burnttoast256            desert256                habiLight                midnight2                professional             summerfruit256           zellner
# busierbee                desert256v2              heliotrope               miko                     proton                   surveyor                 zen
# busybee                  desertEx                 hemisu                   mint                     pspad                    swamplight               zenburn
# buttercream              desertedocean            herald                   mizore                   psql                     symfony                  zenesque
# bvemu                    desertedoceanburnt       herokudoc                mod_tcsoft               putty                    synic                    zephyr
# bw                       detailed                 herokudoc-gvim           molokai                  pw                       tabula                   zmrok
# c                        devbox-dark-256          holokai                  monokain                 pyte                     tango
# cake16                   deveiate                 hornet                   montz                    python                   tango-desert
# 
