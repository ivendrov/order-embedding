

// globals
var db = {};
var datasets_index = [];
var current_methods_dict = [];
var methods = []; // keeps track of currently displayed methods
var cur_method = [];
var dataset_select = [];
var display_select = [];
var max_rank = 1000;
var image_urls;
var caption_texts;


function init() {
    console.log("Loading page");

    d3.json("roc/index.json", function (data) {
        datasets_index = data;
        var datasets = [];
        for (var dataset in datasets_index){
            if (datasets_index.hasOwnProperty(dataset)) {
                datasets.push(dataset);
            }
        }

        dataset_select = d3.select("#data_select").select("[name=dataset]");
        display_select = d3.select("#data_select").select("[name=display]");

        dataset_select.selectAll("option")
            .data(datasets)
            .enter()
            .append("option")
            .attr("name", function(d){
                return d;
            })
            .text(function(d){
                return d;
            });
        dataset_select.on("change",update_methods);
        update_methods();

        // get URL GET params
        var queryDict = {};
        queryDict["method"] = [];
        location.search.substr(1).split("&").forEach(function(item) {
            var split = item.split("=");
            var name = split[0];
            var value = split[1];
            if (name == "method")
                queryDict["method"].push(value);
            else
                queryDict[name] = value
        });
        if (queryDict.hasOwnProperty("dataset") && queryDict.hasOwnProperty("method")){
            console.log("Loading page with form");
            dataset_select.node().value = decodeURIComponent(queryDict["dataset"]);
            display_select.node().value = queryDict["display"];
            console.log(queryDict["display"]);


            update_methods();

            var method_select = d3.select("#data_select").select("[name=method]").node();

            for (var i =0; i < method_select.options.length; i++) {
                var option = method_select.options[i];
                if (queryDict["method"].indexOf(option.value) >= 0) {
                    option.selected = true;
                }
            }
            load();
        }

    });
}

function update_methods(){
    methods_list = datasets_index[dataset_select.node().value];
    methods_list.sort();

    var method_select = d3.select("#data_select").select("[name=method]");
    method_select.selectAll("*").remove();
    method_select
        .selectAll("option")
        .data(methods_list)
        .enter()
        .append("option")
        .attr("name", function(d){
        return d;
        })
        .text(function(d){
            return d;
        });
}
function selectedMethods() {
    var form = document.getElementById("data_select");
    var methods = [];
    // add all selected methods to DB
    for (var i =0; i < form.method.options.length; i++){
        var option = form.method.options[i];
        if (option.selected){
            methods.push(option.value);
        }
    }
    return methods;
}
function selectedDataset() {
    var form = document.getElementById("data_select");
    return form.dataset.value;
}

function load() {
  // populate datasets
    var dataset = selectedDataset();
    var methods = selectedMethods();


    current_methods_dict = datasets_index[dataset];

    datasets_loaded = 0;
    for (var i = 0; i < methods.length; i++){
        var method = methods[i];
        var path = 'roc/' + dataset + '/' + method;
        console.log("Loading dataset from " + path);
        loadDataset(path, method, methods.length);

    }

    // load captions and images

    d3.json('roc/' + dataset + '/captions.json', function(data) {
        caption_texts = data;
        console.log("Loaded caption info");
    });
    d3.json('roc/' + dataset + '/image_urls.json', function(data) {
        image_urls = data;
        console.log("Loaded image info");
    });


    document.getElementById("dataset_load_error").innerHTML = "";
    displayResults()

}

/** display the caption, given a caption result dict */
function displaySentence(caption){
    var caption_id = caption.id;


    // now display the images
    var gt_image_json = caption.gt_image;
    var images_json = caption.top_images;


    d3.select("#images").selectAll("figure").remove();

    var images = d3.select("#images")
        .selectAll("figure")
        .data([gt_image_json].concat(images_json));

    // add new figures
    var newFigs = images
        .enter()
        .append("figure")
        .attr("class", "thumbnail");

    // add new images
    newFigs
        .append("img");

    // add new captions
    newFigs
        .append("figcaption")
        .text(function(d,i){
            var label = "";
            if (i == 0) label = "GT";
            else label = "#" + i.toString();
            return label + ", score = " + d3.format(".3f")(d.score);
        });

    // update existing images
    images
        .select("img")
        .style("border-color", function(d){
            if (d.id == caption.gt_image.id){
                return "green"
            } else {
                return "red"
            }
        })
        .attr("src", function(d){
            return image_urls[d.id];
    });

}

/** display the table of statistics */
function displayStats(colors){
    console.log("Generating stats");
    var statsTable = d3.select("#stats_table");
    statsTable.selectAll("*").remove();

    // generate data
    var methods = Object.keys(db);
    var stat_names = ["Method"];
    var data = [];
    var best_value = [];


    var data_names = Object.keys(db[methods[0]].stats);
    data_names.sort();

    for (var i = 0; i < data_names.length; i++){
        var stat = data_names[i];
        stat_names.push(stat);
        if (stat == "median_rank" || stat == "mean_rank")
            best_value.push(1000000);
        else
            best_value.push(0);
    }
    for (method in db){

        var stats = db[method].stats;
        var method_data = [method];

        // compute best values for all statistics
        for (var i = 0; i < data_names.length; i++){
            var name = data_names[i];
            var value = stats[name];
            method_data.push(value)
            if (name == "median_rank" || name == "mean_rank") {
                if (value < best_value[i]) {
                    best_value[i] = value
                }
            } else {
                if (value > best_value[i]) {
                    best_value[i] = value;

                }

            }
        }
        data.push(method_data);
    }


    // table header
    statsTable.append("thead")
        .append("tr")
        .selectAll("th")
        .data(stat_names)
        .enter()
        .append("th")
        .text(function (d) {
            return d;
        });


    // table body
    var tbody = statsTable.append("tbody");
    var rows = tbody.selectAll("tr")
                    .data(data)
                    .enter()
                    .append("tr");

    rows.selectAll("td")
        .data(function (d) {
            return d;
        })
        .enter()
        .append("td")
        .style("color", function(d, i){
            if (i == 0){
                var color = colors[methods.indexOf(d)];
                return color;
            } else {
                return null;
            }
        })
        .style("text-align", "center")
        .html(function(d, i){
            var text = [];
            if (i == 0){
                if (display_select.node().value == 'ranks') {
                    return "<b>" + d + "</b>";
                } else {
                    return "<b>" + d + "</b>";
                }
            } else if (i == 1) {
                text = d3.format(" >4g")(d);
            } else {
                text = d3.format(".3f")(d);
            }
            if (best_value[i-1] == d.value){
                text = "<u>" + text + "</u>";
            }
            return text;

        });


}

function displayResults() {
    // display caption search form
    d3.select("#caption_search").style("display", "block");
    // generate random colors for each method
    var colors = [];
    methods = [];
    for (method in db){
        colors.push(d3.rgb('#'+Math.random().toString(16).substr(-6)).darker(1));
        methods.push(method)
    }

    displayStats(colors);

    // display hyperparams
    d3.select("#hyperparams").selectAll("div").remove();
    d3.select("#hyperparams").selectAll("div")
        .data(methods)
        .enter()
        .append("div")
        .style("color", function(d, i){
            return colors[i];
        })
        .append("pre")
        .text(function(d){
            return JSON.stringify(db[method].hyperparams, null, 2);
        });


    var datasets = [];
    var sorted_datasets = [];
    for (method in db) {
        var dset = db[method].sentences;
        datasets.push(dset);
        var sorted = dset.slice(0).sort(function (a, b) {
            return a.rank - b.rank
        });
        sorted_datasets.push(sorted);
    }

    var charts;
    if (datasets.length == 2 && display_select.node().value == 'ranks'){
        d3.select("#compare_chart")
        .append("h2")
        .text("Sorted in order of increasing difference");
        charts = d3.selectAll(".chart_div");
    } else {
        charts = d3.select("#roc_chart");
    }
    charts = charts.append("svg");





    // Mike's margin convention
    var margin = {top: 20, right: 10, bottom: 20, left: 40};

    var w = 500 - margin.left - margin.right,
        h = 500 - margin.top - margin.bottom;

    charts
        .attr("width", w + margin.left + margin.right)
        .attr("height", h + margin.top + margin.bottom);
    charts.selectAll("*").remove();

    var svgs = charts.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("width", w)
        .attr("height", h)
        .attr("class", "chart");

    svgs.append("rect")
        .attr("width",w)
        .attr("height",h)
        .style("fill", d3.rgb(240,240,240));




    var dataset1 = datasets[0];

    console.log(display_select.node().value);

    if (display_select.node().value == 'ranks') {
        var xScale = d3.scale.linear()
            .domain([1, dataset1.length - 1])
            .range([0, w]);

        var yScale = d3.scale.log()
            .domain([1, max_rank /*d3.max(dataset1, get_rank)*/])
            .range([0, h]);
    } else {
        var xScale = d3.scale.linear()
            .domain([0, 1])
            .range([0, w]);

        var yScale = d3.scale.linear()
            .domain([1/max_rank, 1])
            .range([h, 0]);
    }




    addDatasets(sorted_datasets, d3.select("#roc_chart .chart"), xScale, yScale, colors);

    

    // draw axes
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");

    var xlabel = svgs.append("text")
        .attr("text-anchor", "end")
        .attr("x", w - 30)
        .attr("y", h - 6);


    var ylabel = svgs.append("text")
        .attr("text-anchor", "end")
        .attr("x", -30)
        .attr("y", 6)
        .attr("dy", ".75em")
        .attr("transform", "rotate(-90)");


    if (display_select.node().value == 'ranks') {
        yAxis.tickFormat(d3.format("d"));
        xlabel.text("sentence index");
        ylabel.text("rank of GT image");
    } else {
        xlabel.text("recall");
        ylabel.text("precision");
    }



    svgs.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + h + ")")
        .call(xAxis);

    svgs.append("g")
        .attr("class", "axis")
        .call(yAxis);



    // if there are just 2 datasets and in Rank Mode, we display the comparing chart
    if (datasets.length == 2 && display_select.node().value == 'ranks') {

        var dset1 = datasets[0];
        var dset2 = datasets[1];
        console.log(dset1.length);
        console.log(dset2.length);
        if (dset1.length != dset2.length) return;
        var zipped = _.zip(dset1, dset2);
        var sorted_zipped = zipped.sort(function(a, b) {
            return (a[0].rank - a[1].rank) - (b[0].rank - b[1].rank);
        });
        var sorted = _.zip.apply(null, sorted_zipped);

        var compareYScale = d3.scale.linear()
            .domain([1, max_rank])
            .range([0, h]);

        addDatasets(sorted, d3.select("#compare_chart .chart"), xScale, compareYScale, colors);

    }

}

function addDatasets(datasets, svg, xScale, yScale, colors){




    var currentlySelected = null;

    var roc_curves = svg.selectAll(".roc_curve")
        .data(datasets)
        .enter()
        .append("g")
        .attr("class", "roc_curve")
        .style("fill", function(d, i) {
            return colors[i]
        });

    var datapoints = roc_curves.selectAll("circle")
        .data(function(d) {
            return d;
        })
        .enter()
        .append("circle")
        .attr("class", "roc_point")
        .attr("cx", function (d, i) {
            if (display_select.node().value == 'ranks')
                return xScale(i);
            else {
                return xScale(i / datasets[0].length);

            }
        })
        .attr("cy", function (d) {
            if (display_select.node().value == 'ranks')
                return yScale(d.rank);
            else {
                return yScale(1 - (d.rank - 1) / max_rank);

            }
        });
    if (display_select.node().value == 'ranks') {
        datapoints.attr("r", 3)
            .on("mouseover", function (d) {
                if (this != currentlySelected)
                    d3.select(this).attr("r", 8).style("fill", d3.rgb("yellow"));
            })
            .on("mouseout", function (d) {
                if (this != currentlySelected)
                    d3.select(this).attr("r", 3).style("fill", null);

            })
            .on("click", function (d, _, method_id) {
                if (currentlySelected) {
                    d3.select(currentlySelected).attr("r", 3).style("fill", null);
                }
                d3.select(this).attr("r", 8).style("fill", d3.rgb("green"));
                currentlySelected = this;
                loadSentence(d.id, method_id);
            });
    } else {
       datapoints.attr("r", 1)
        .on("click", function (d, _, method_id) {
                if (currentlySelected) {
                    d3.select(currentlySelected).attr("r", 1).style("fill", null);
                }
                d3.select(this).attr("r", 8).style("fill", d3.rgb("green"));
                currentlySelected = this;
                loadSentence(d.id, method_id);
            });
    }



}

function loadSentence(caption_id, method_id){
    cur_method = methods[method_id];
    var sentence = db[cur_method].sentences[caption_id];
    rank = sentence.rank;

    d3.select("#sentence_id").text("ID: " + caption_id.toString() + "; rank = " + rank.toString());
    d3.select("#sentence_details").select("h1").text(caption_texts[caption_id.toString()]);
    displaySentence(sentence)
}

function captionSearch(){
    var form = document.getElementById("data_select");
    var dataset = form.dataset.value;
    var method = form.method.value;
    var caption_id = document.getElementById("caption_search").caption_id.value;

    loadSentence(parseInt(caption_id), 0);
    return false;
}

// Data Loading
function loadDataset(jsonpath, name) {
    var jsonmod = jsonpath + '.json';
    console.log(jsonmod);
    $.ajax({
      dataType: "json",
      url: jsonmod,
        async: false,
      success: function (data) {
          if (data == null) {
              document.getElementById("dataset_load_error").innerHTML = "Error: data not found for " + name;
          } else {
              db[name] = data; // assign to global
          }
      }

    });
}
