# coding: utf-8
import model, tools, evaluation, datasets, hierarchy_data
dataset = datasets.load_dataset('snli')
dataset['train']
m = tools.load_model('snapshots/entailment_symmetric.npz')
m['h_error']
dev = hierarchy_data.HierarchyData(dataset['dev'], m['worddict'], n_words=m['n_words'])
dev = hierarchy_data.HierarchyData(dataset['dev'], m['worddict'], n_words=len(m['worddict']))
test = hierarchy_data.HierarchyData(dataset['test'], m['worddict'], n_words=len(m['worddict']))
dev_caps, dev_target = dev.all()
dev_s = tools.encode_sentences(m, dev_caps)
dev_errs = h_error(dev_s)
dev_errs = m['h_error'](dev_s)
eval_accuracy(dev_errs, dev_target, dev_errs, dev_target)
evaluation.eval_accuracy(dev_errs, dev_target, dev_errs, dev_target)
test_caps, test_target = test.all()
test_s = tools.encode_sentences(m, test_caps)
test_errs = model['h_error'](test_s)
test_errs = m['h_error'](test_s)
eval_accuracy(dev_errs, dev_target, test_errs, test_target)
evaluation.eval_accuracy(dev_errs, dev_target, test_errs, test_target)
evaluation.eval_accuracy(test_errs, test_target, test_errs, test_target)
