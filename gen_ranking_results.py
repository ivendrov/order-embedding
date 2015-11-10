import tools, evaluation

def run(model_loc, split='dev'):
    model = tools.load_model(model_loc)
    res = evaluation.evalrank(model, split)

    print ' & '.join(res)


