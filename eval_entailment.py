# coding: utf-8
import model, tools, evaluation, datasets, hierarchy_data
dataset = datasets.load_dataset('snli')
dataset['train']
m = tools.load_model('your_snapshots_directory/model_name')
dev = hierarchy_data.HierarchyData(dataset['dev'], m['worddict'], n_words=len(m['worddict']))
test = hierarchy_data.HierarchyData(dataset['test'], m['worddict'], n_words=len(m['worddict']))
dev_caps, dev_target = dev.all()
dev_s = tools.encode_sentences(m, dev_caps)
dev_errs = m['h_error'](dev_s)
print('Dev accuracy: ' + str(evaluation.eval_accuracy(dev_errs, dev_target, dev_errs, dev_target)[0]))
test_caps, test_target = test.all()
test_s = tools.encode_sentences(m, test_caps)
test_errs = m['h_error'](test_s)
print('Test accuracy: ' + str(evaluation.eval_accuracy(test_errs, test_target, test_errs, test_target)[0]))
