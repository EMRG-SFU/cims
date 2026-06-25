-- watermark.lua — toggleable DRAFT watermark for the CIMS book.
--
-- Off by default. Turns on when the `watermark` metadata flag is true,
-- which is set by the `watermark` profile (_quarto-watermark.yml). Render:
--
-- NB: the flag is `watermark`, not `draft`. `draft` is a reserved Quarto
-- metadata key (Quarto's own draft-page feature) and setting it breaks
-- book navigation, so we use our own key.
--   quarto render --profile watermark   (watermarked)
--   quarto render                       (clean)
--
-- Format-aware: emits LaTeX (draftwatermark package) for PDF and a CSS
-- overlay for HTML. The injected markup is appended to header-includes,
-- so it coexists with the existing pdf header-includes in _quarto.yml.
--
-- To change the text/appearance, edit the two blocks below.

local WATERMARK_TEXT = "DRAFT"

-- LaTeX: draftwatermark works with xelatex + KOMA scrbook. Lightness
-- 0.9 keeps it a faint grey behind the text on every page.
local function latex_watermark()
  return table.concat({
    "\\usepackage{draftwatermark}",
    "\\SetWatermarkText{" .. WATERMARK_TEXT .. "}",
    "\\SetWatermarkScale{1}",
    "\\SetWatermarkLightness{0.9}",
  }, "\n")
end

-- HTML: a single fixed, diagonal, non-interactive overlay drawn via
-- body::before so no DOM element is needed. Faint red, sits behind
-- nothing (pointer-events: none) and is excluded from selection.
local function html_watermark()
  return [[
<style>
body::before {
  content: "]] .. WATERMARK_TEXT .. [[";
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-45deg);
  font-size: 12vw;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: rgba(190, 0, 0, 0.06);
  white-space: nowrap;
  pointer-events: none;
  user-select: none;
  -webkit-user-select: none;
  z-index: 9999;
}
</style>
]]
end

function Meta(meta)
  -- Only act when watermark mode is explicitly on.
  if not meta.watermark then
    return nil
  end

  local block
  if FORMAT:match("latex") then
    block = pandoc.RawBlock("latex", latex_watermark())
  elseif FORMAT:match("html") then
    block = pandoc.RawBlock("html", html_watermark())
  else
    return nil
  end

  -- Append to header-includes, preserving anything already there.
  local includes = meta["header-includes"]
  if includes == nil then
    includes = pandoc.MetaList({})
  elseif includes.t ~= "MetaList" then
    includes = pandoc.MetaList({ includes })
  end
  includes:insert(pandoc.MetaBlocks({ block }))
  meta["header-includes"] = includes

  return meta
end
