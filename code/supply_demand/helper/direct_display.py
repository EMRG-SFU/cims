from IPython.display import HTML
import random

def toggled():
    return(HTML('''<script>
    code_show=true; 
    function code_toggle() {
    if (code_show){
        $('div.input').hide();
    } 
    else {
        $('div.input').show();
    }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    The code cells for this Jupyter notebook are hidden for easier reading.
    To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.'''))




def one_cell_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Click here to hide/show the current code cell'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)    