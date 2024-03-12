# ------------------------------------------------------------------------------
# Based on spacy's download code
# https://github.com/explosion/spaCy/blob/master/spacy/cli/download.py
# ------------------------------------------------------------------------------
import requests
import tqdm

from packaging.version import InvalidVersion, Version
from urllib.parse import urljoin
from typing import Optional
from pathlib import Path
from wasabi import msg

from . import about


def download_model(out_path:str, token:str=None, model_name:str="all_models", overwrite:bool=False) -> None:
    # determine what the compatible version of the data models are 
    all_compatible_models = get_compatibility(token=token)
    model_version = get_model_version(model_name, all_compatible_models)
    release_id = get_release_id(model_version, token=token)
    download_release_asset(out_path, release_id, model_name, model_version,
                           token=token, overwrite=overwrite)

def get_compatibility(token:str=None) -> dict:
    version = get_minor_version(about.__version__)

    headers = {}
    if token is not None:
        headers.update({'Authorization': f"token {token}"})

    r = requests.get(about.__compatibility__, 
                     headers=headers)

    if r.status_code != 200:
        msg.fail(
            f"Server error ({r.status_code})",
            f"Couldn't fetch compatibility table. Please find a package for your CIMS "
            f"installation ({about.__version__}), and download it manually.",
            exits=1,
        )
    comp_table = r.json()
    comp = comp_table["CIMS"]
    if version in comp:
        return comp[version]
    elif f"v{version}" in comp:
        return comp[f"v{version}"]
    else:
        msg.fail(f"No compatible packages found for v{version} of CIMS", exits=1)

def get_minor_version(version:str) -> Optional[str]:
    """Get the major + minor version (without patch or prerelease identifiers).
    From https://github.com/explosion/spaCy/blob/master/spacy/util.py

    Parameters
    ----------
    version : str
        The version.

    Returns
    -------
    Optional[str]
        The major + minor version or None if version is invalid. 
    """
    try:
        v = Version(version)
    except (TypeError, InvalidVersion):
        return None
    return f"{v.major}.{v.minor}"

def get_model_version(model_name:str, all_compatible_models:dict) -> str:
    if model_name not in all_compatible_models:
        msg.fail(
            f"No compatible package found for '{model_name}' (CIMS v{about.__version__})",
            exits=1,
        )
    return all_compatible_models[model_name][0]

def get_asset_name(model_name: str, version: str) -> str:    
    filename = f"{model_name}-{version}.tar.gz"
    return filename

def get_release_id(model_version:str, token: str) -> str:    
    release_tag = f"{model_version}"
    # urljoin requires path ends with /, or the last path part will be dropped
    base_url = about.__download_url__
    if not base_url.endswith("/"):
        base_url = about.__download_url__ + "/"
    release_url = urljoin(base_url, f"tags/{release_tag}")

    headers = {"Accept": "application/vnd.github+json"}
    if token is not None:
        headers.update({'Authorization': f"token {token}"})
    
    r = requests.get(release_url, headers=headers)
    release_id = r.json()['id']
    return release_id

def download_release_asset(out_path:str, release_id: int, model_name, model_version, token: str=None, overwrite: bool=False) -> None:
    # Asset Name
    asset_name = get_asset_name(model_name, model_version)

    # urljoin requires path ends with /, or the last path part will be dropped
    base_url = about.__download_url__
    if not base_url.endswith("/"):
        base_url = about.__download_url__ + "/"
    
    # Find Release Asset
    asset_list_url = urljoin(base_url, f"{release_id}/assets")     
        
    headers = {"Accept": "application/vnd.github+json"}
    if token is not None:
        headers.update({'Authorization': f"token {token}"})
    assets = requests.get(asset_list_url, headers=headers)

    matching_assets = [a['id'] for a in assets.json() if a['name']==asset_name]
    if len(matching_assets) > 1:
        msg.fail(
                title="Multiple model assets",
                text=f"There were multiple '{asset_name}' files available for download. Check release.",
                exits=True,
            )    
    else:
        asset_id = matching_assets[0]

    asset_url = urljoin(base_url, f"assets/{asset_id}")
    headers = {"Accept": "application/octet-stream"}
    if token is not None:
        headers.update({'Authorization': f"token {token}"})
    response = requests.get(asset_url, headers=headers, stream=True)

    if response.status_code == 200:
        out_file = Path(out_path).joinpath(asset_name)
        write_file(response, out_file, overwrite=overwrite)
    
    return out_file

def write_file(response:requests.Response, out_file:str, overwrite=False):
    if (not overwrite) and Path(out_file).is_file():
        msg.fail(
                title="File exists.",
                text=f"{out_file} already exists. Use `overwrite=True` to allow overwrites.",
                exits=True,
            )    
    total = int(response.headers.get('Content-Length', 0))
    with open(out_file, 'wb') as file, tqdm.tqdm(desc=out_file, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                size = file.write(chunk)
                bar.update(size)
