import plotly.graph_objects as go

def _legend_required_height(fig, line_factor=1.25, pad_px=8, verbose: bool = False):
    """
    Roughly estimate how many vertical pixels the legend needs.
    """
    n_items = len(fig.data)                 # one entry per trace
    if verbose:
        print(f"How many legend items: {n_items}")
    # Font size may be None → Plotly falls back to 12 pt
    font_sz = getattr(fig.layout.legend.font, "size", None) or 12
    if verbose:
        print(f"font_sz: {font_sz}")
    # Convert pt → pixel (1 pt ≈ 1.333 px)
    font_px = font_sz * 1.333
    if verbose:
        print(f"font_px: {font_px}")
    # Height of a single legend row (font + line spacing)
    row_h = font_px * line_factor
    if verbose:
        print(f"row_h: {row_h}")
    # Total height = rows + top/bottom padding
    return n_items * row_h + 2 * pad_px


def ensure_legend_fits(fig,
                       max_iter: int = 100,
                       height_step: int = 40,
                       margin_step: int = 20,
                       grow_margin: bool = True,
                       verbose: bool = False):
    """
    Increase figure height (and optionally bottom margin) until the legend
    fits inside the drawable area.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure you want to protect.
    max_iter : int
        Safety guard – stop after this many enlargements.
    height_step : int
        How many pixels to add to `fig.layout.height` each iteration.
    margin_step : int
        How many pixels to add to the bottom margin (if `grow_margin=True`).
    grow_margin : bool
        If True, we first try to give the legend more breathing room by
        expanding the bottom margin before we increase the whole canvas.
    verbose : bool
        Print diagnostic info.
    """
    # Make sure we have explicit height/margins – otherwise Plotly autosizes.
    if fig.layout.height is None:
        fig.update_layout(height=400)          # sensible default
    if fig.layout.margin is None:
        #fig.update_layout(margin=dict(t=80, b=80, l=80, r=80))
        fig.update_layout(margin=dict(t=2, b=2, l=2, r=2))

    for i in range(max_iter):
        # ----- 1️⃣  Compute available vertical space -----------------
        total_h = fig.layout.height
        margin = fig.layout.margin
        avail_h = total_h - (margin.t or 0) - (margin.b or 0)

        # ----- 2️⃣  Estimate legend height ---------------------------
        if verbose:
            req_h = _legend_required_height(fig, verbose=True)
        else:
            req_h = _legend_required_height(fig)

        if verbose:
            print(f"[iter {i}] total={total_h}px, "
                  f"avail={avail_h:.0f}px, legend_req={req_h:.0f}px")

        # ----- 3️⃣  Does it fit? ------------------------------------
        if req_h <= avail_h:
            if verbose:
                print("✅ Legend fits – stop resizing.")
            break   # success!

        # ----- 4️⃣  Not enough room – enlarge ------------------------
        if grow_margin and (margin.b or 0) < total_h * 0.3:
            # First try to give the legend more bottom margin.
            new_bottom = (margin.b or 0) + margin_step
            fig.update_layout(margin=dict(b=new_bottom))
            if verbose:
                print(f"   → Adding {margin_step}px to bottom margin "
                      f"(now {new_bottom}px).")
        else:
            # If margin is already generous, bump the whole canvas.
            new_h = total_h + height_step
            fig.update_layout(height=new_h)
            if verbose:
                print(f"   → Raising figure height by {height_step}px "
                      f"(now {new_h}px).")
    else:
        if verbose:
            print("⚠️  Reached max_iter without fully fitting legend.")
    return fig
