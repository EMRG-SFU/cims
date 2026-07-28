
// LUMO came up with this function. It required a tiny bit of tweaking, but was good overall.
function tableToObject(table) {
  const data = {};

  // Grab header cells (skip the very first corner cell)
  const colHeaders = Array.from(
    table.querySelectorAll('thead th:not(:first-child)')
  ).map(th => th.textContent.trim());

  // Iterate over each body row
  table.querySelectorAll('tbody tr').forEach(row => {
    // First cell of the row is the row name (usually a <th>)
    // I guess the `querySelector` vs `querySelectorAll` just grabs the first child, rather than all of them as 
    // we do below.
    const rowHeader = row.querySelector('th, td');
    const rowName = rowHeader.textContent.trim();

    // Prepare a sub‑object for this row
    data[rowName] = {};

    // All remaining cells are the data cells. The `slice(1)` chops off the first
    // of the values, which in our case is the row name/id. If the data this function processes
    // ever changes, such that the row ID is a `<th>` element while the regular values remain
    // `<td>` elements, this will have to change.
    const cells = Array.from(row.querySelectorAll('td')).slice(1);

    cells.forEach((cell, idx) => {
      const colName = colHeaders[idx];               // match column header
      const cellValue = cell.textContent.trim();     // raw string; convert if needed
      data[rowName][colName] = cellValue;
    });
  });

  return data;
}

// Usage
// const tbl = document.getElementById('myTable');
// const result = tableToObject(tbl);
// console.log(result);

export { tableToObject }
