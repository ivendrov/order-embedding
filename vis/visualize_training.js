function init() {
    console.log("Loading page");

    var model_select = d3.select("#model_select").select("[name=model]");

    d3.json("training/index.json", function (data) {

        update_select(model_select, data);



        // get URL GET params
        var queryDict = {};

        location.search.substr(1).split("&").forEach(function(item) {
            var split = item.split("=");
            var name = split[0];
            var value = split[1];
            if (name == "model") {
                if (!queryDict.hasOwnProperty("model")) {
                    queryDict["model"] = []
                }
                queryDict["model"].push(value);
            }
            else
                queryDict[name] = value
        });
        if (queryDict.hasOwnProperty("model")){
            console.log("Loading page with form");

            var options = model_select.node().options;
            for (var i =0; i < options.length; i++) {
                var option = options[i];
                if (queryDict["model"].indexOf(option.value) >= 0) {
                    option.selected = true;
                    console.log("Selected model " + option.value)
                }
            }
            loadModels(queryDict["model"]);
        }

    });
}

/**
 * Populates the given form selector with the given string options
 */
function update_select(formSelector, options){
    options.sort();

    formSelector.selectAll("*").remove();
    formSelector
        .selectAll("option")
        .data(options)
        .enter()
        .append("option")
        .attr("name", function(d){
        return d;
        })
        .text(function(d){
            return d;
        });
}

// two global variables storing all loaded models and their colors
models = [];
colors = [];
/**
 * Loads the given models into memory
 * @param modelnames model names
 * @param callback function to call with resulting array of json logs
 */
function loadModels(modelnames, callback){
    models = [];

    modelnames.forEach(function(modelname){
        d3.json('training/' + modelname + '.json', function(data){
            models.push(data);
            if (models.length == modelnames.length)
                displayModels()
        });
    });
}



function displayData(e) {
    var dataSelect = e.srcElement;
    var data_streams = [];
    var options = dataSelect.options;
    for (var i = 0; i < options.length; i++){
        if (options[i].selected){
            console.log(options[i].value);
            data_streams.push(options[i].value)
        }
    }

    // a chart is an array of data streams (usually just one)
    // display the Training analogue of each data stream, if it exists
    var charts = [];
    data_streams.forEach(function(dataName){
        if (_.every(models, function(model) { return model.data.hasOwnProperty("Training" + dataName)})){
            charts.push([dataName, "Training" + dataName]);
        } else {
            charts.push([dataName]);
        }
    });

    // get minimal and maximal values for each chart
    var x_maximum = Number.NEGATIVE_INFINITY;
    var y_minima = new Array(charts.length);
    var y_maxima = new Array(charts.length);
    for (var i = 0; i < charts.length; i++){
        y_minima[i] = Number.POSITIVE_INFINITY;
        y_maxima[i] = Number.NEGATIVE_INFINITY;
        charts[i].forEach(function(data_stream) {
            for (var j = 0; j < models.length; j++) {
                if (models[j].data.hasOwnProperty(data_stream)) {
                    models[j].data[data_stream].forEach(function (p) {
                        x_maximum = Math.max(x_maximum, p.x);
                        y_maxima[i] = Math.max(y_maxima[i], p.y);
                        y_minima[i] = Math.min(y_minima[i], p.y);
                    });
                }
            }
        });
    }





    var charts_main = d3.select("#charts");

    charts_main.selectAll("div").remove();

    // add new charts if need be
    var new_divs = charts_main.selectAll("div")
        .data(charts)
        .enter()
        .append("div");
    // add headers to new charts
    new_divs
        .append("h1")
        .text(function(d) {
            if (d.length == 1) {
                return d[0]
            } else {
                return d[0] + " (dashed = Training)"
            }
        });

    // add svgs to new charts

    var margin = {top: 20, right: 30, bottom: 50, left: 60}; // Mike's margin convention

    var w = 1000 - margin.left - margin.right,
        h = 500 - margin.top - margin.bottom;

    new_divs
        .append("svg")
        .attr("width", w + margin.left + margin.right)
        .attr("height", h + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("width", w)
        .attr("height", h)
        .attr("class", "chart");

    // compute chart scales
    var xScales = [];
    var yScales = [];
    for (var i = 0; i < data_streams.length; i++) {
        xScales.push(d3.scale.linear()
            .domain([0, x_maximum])
            .range([0, w]));

        var yScale = d3.scale.log();
        if (y_minima[i] <= 0 || y_maxima[i] / y_minima[i] < 10 ){
            yScale = d3.scale.linear();
        }
        yScales.push(yScale
            .domain([y_minima[i], y_maxima[i]])
            .range([h, 0]));
    }


    // Add all needed curves to all charts
    var datastreams =
        d3.selectAll(".chart")
            .selectAll(".datastream") // all datastreams for this chart
            .data(function(d) { return d;})
            .enter()
            .append("g")
            .attr("class", "datastream");

    // make training info dashed
    datastreams
        .filter(function (d,i) { return i == 1})
        .style("stroke-dasharray", ("3, 3"))
        .style("opacity", 0.5);


    datastreams
        .selectAll(".curve") // all curves for this datastream
            .data(function (d) { return models.map(function (model) {return model.data[d];}); })
            .enter()
            .append("path")
            .attr("class", "curve")
            .style("stroke", function(d, i, datastream_index) {
                return colors[i]
            });

    /**
     * @param chart_index index of chart
     * @returns a function which, given an array of points, returns the SVG path string, adjusted to the chart's scale
     */
    function path_string(chart_index) {
        return d3.svg.line()
            .x(function (d) {
                return xScales[chart_index](d.x)
            })
            .y(function (d) {
                return yScales[chart_index](d.y)
            })
    }

    // Populate the doubly-nested selection of curves
    var curves = d3.selectAll(".chart").selectAll(".curve");
    curves
        .attr("d", function(d, model_index, chart_index){
            return path_string(chart_index)(d);
        });

    // x label
    d3.selectAll(".chart")
        .append("text")
        .attr("text-anchor", "center")
        .attr("x", w/2)
        .attr("y", h + 40)
        .text(models[0].xLabel);

    // y label
    d3.selectAll(".chart")
        .append("text")
        .attr("text-anchor", "center")
        .attr("x", -h/2)
        .attr("y", -60)
        .attr("dy", ".75em")
        .attr("transform", "rotate(-90)")
        .text(function(d) { return d[0]; });




    // draw axes
    d3.selectAll(".chart")
        .append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + h + ")")
        .each(function(d, i) {
            d3.svg.axis()
                .scale(xScales[i])
                .orient("bottom")
                .tickSize(-h, 0, 0)
                .ticks(10)
            (d3.select(this));
        });

    d3.selectAll(".chart")
        .append("g")
        .attr("class", "axis")
        .each(function(d, i){
             d3.svg.axis()
                .scale(yScales[i])
                .orient("left")
                .ticks(10)
             (d3.select(this));
        });
}


/**
 * Randomly assigns colors to models, outputs hyperparameters, and a list of possible data sources
 */
function displayModels(){
    console.log("Displaying ");


    // generate random colors for each method
    colors = [];
    for (var i = 0; i < models.length; i++){
        colors.push(d3.rgb('#'+Math.random().toString(16).substr(-6)).darker(1));
    }

    // display hyperparams
    d3.select("#hyperparams").selectAll("div").remove();
    var hyperDivs = d3.select("#hyperparams").selectAll("div")
        .data(models)
        .enter()
        .append("div")
        .style("color", function(d, i){
            return colors[i];
        });
    hyperDivs
        .append("h2")
        .append("b")
        .text(function(d){
            return d.name;
        });
    hyperDivs
        .append("pre")
        .text(function(d){
            return JSON.stringify(d.hyperparams, null, 2);
        });


    // populate data selector



    function stripTraining(str){
        if (str.startsWith("Training"))
            return str.substring("Training".length);
        else
            return str;
    }

    var all_data_names = models.map(function(model){
        return _.uniq(_.keys(model.data).map(stripTraining).sort(), true);
    });
    var data_names = _.intersection.apply(null, all_data_names);

    var data_selector = d3.select("#data_select").select("[name=data]").node();
    data_selector.options.length = 0;
    data_names.forEach(function(dataName,i){
        data_selector.options[i] = new Option(dataName, dataName);
    })
    data_selector.addEventListener("change", displayData, false)
}