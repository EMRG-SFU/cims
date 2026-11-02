-- tech-block.lua: PDF styling for the technology blocks in the sector
-- chapters (fenced divs with the .tech-block class).
--
-- HTML gets its indent and styling from style.css. For LaTeX, this filter
-- indents the block (addmargin, from KOMA-Script) and shifts its longtable
-- by the same amount, sets the table in a smaller size, and turns the
-- block's first paragraph ("Technologies") into a small sans-serif label.
-- Other formats pass through unchanged.

local INDENT = "1.75em"

function Div(el)
  if not el.classes:includes("tech-block") or not quarto.doc.is_format("latex") then
    return nil
  end

  local blocks = pandoc.List()
  blocks:insert(pandoc.RawBlock("latex",
    "\\begin{addmargin}[" .. INDENT .. "]{0pt}\\begingroup" ..
    "\\setlength{\\LTleft}{" .. INDENT .. "}\\setlength{\\LTright}{0pt}\\small"))

  for i, block in ipairs(el.content) do
    if i == 1 and block.t == "Para" then
      blocks:insert(pandoc.RawBlock("latex",
        "\\noindent{\\sffamily\\bfseries\\footnotesize\\color{cimsblue}\\MakeUppercase{" ..
        pandoc.utils.stringify(block) .. "}}\\par\\nopagebreak\\smallskip"))
    else
      blocks:insert(block)
    end
  end

  blocks:insert(pandoc.RawBlock("latex", "\\endgroup\\end{addmargin}"))
  return blocks
end
