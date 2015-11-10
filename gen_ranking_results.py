import tools, evaluation

def run(model_loc, split='dev'):
    model = tools.load_model(model_loc)
    res = evaluation.evalrank(model, split)
    def toStr(f):
        return '%.1f' % f

    print ' & '.join(map(toStr, list(res)))


