-- theme-images.lua: keep only the light version of theme-paired images
-- outside HTML.
--
-- Figures such as the sector flowcharts come as a pair of images with the
-- .light-content and .dark-content classes; in HTML, Quarto shows the one
-- that matches the reader's theme. Other formats (PDF, Word) would print
-- both, so this filter removes the .dark-content image there.

function Image(el)
  if el.classes:includes("dark-content") and not quarto.doc.is_format("html") then
    return {}
  end
  return nil
end
