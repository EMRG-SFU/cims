-- strip-chapter-refs.lua
--
-- Quarto auto-injects a "References" section (a level-2 Header followed
-- by a Div with id "refs") at the end of every chapter that declares a
-- `bibliography:` in its frontmatter. In HTML this gives us per-chapter
-- reference lists, which we want. In PDF we want a single combined
-- References section at the end of the book (placed by references.qmd
-- via a manual \printbibliographyORIG call), so the per-chapter
-- injections must be stripped.
--
-- This filter is registered under format.pdf.filters in _quarto.yml, so
-- it only runs during PDF/LaTeX rendering. The level >= 2 check on the
-- Header guard protects the chapter-level title of references.qmd
-- (a level-1 Header) from being stripped along with the per-chapter
-- section-level "References" headings.

local function is_refs_id(id)
  if not id or id == "" then return false end
  return id == "refs" or id:match("^refs[%-_]")
end

local function is_references_heading_id(id)
  if not id or id == "" then return false end
  return id == "references" or id:match("^references%-")
end

function Div(el)
  if is_refs_id(el.identifier) then
    return {}
  end
end

function Header(el)
  if el.level >= 2 and is_references_heading_id(el.identifier) then
    return {}
  end
end
