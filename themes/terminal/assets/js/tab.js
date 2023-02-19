window.switchTab = function(id, index) {
  let tabs = document.getElementById("tabs-" + id).children;
  for (let i = 0; i < tabs.length; i++) {
    tabs[i].className = tabs[i].className.replace("active", "");
  }

  let contents = document.getElementById("tabcontent-" + id).children;
  for (let i = 0; i < contents.length; i++) {
    contents[i].style.display = "none";
  }

  tabs[index].className += " active";
  contents[index].style.display = "block";
}
