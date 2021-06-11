import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import math
################################################################################UNUSED IN COLAB
from data_loader import load_data
from models import SVHF_Net, RandomNet
###############################################################################################


# updates the "correct_mrg" dictionary that maps a speaker's identifier integer
# into a 2-list containing, in order, the number of correct tests that involved
# the speaker and the total number of tests involving the speaker.
# The information is updated based on the given 'correct' tensor that holds
# the results of the predictions (compared with their labels) and the 'speakers'
# tensor that holds the identifiers of the speakers involved in the predictions
def update_marginal_stats(correct_mrg: dict, correct: torch.tensor, speakers: torch.tensor):
  assert correct.shape == speakers.shape
  for res, speaker in zip(correct, speakers):
    speaker = int(speaker)
    if speaker in correct_mrg:
      correct_mrg[speaker][0] += res
      correct_mrg[speaker][1] += 1 
    else:
      correct_mrg[speaker] = [res, 1]


def train(model, device, train_loader, loss_f, acc_funcs: list, optimizer, 
          epoch: int, timeout: int):
  model.train()
  correct_cnt = 0
  correct_mrg = dict()
  start_time = time.time()
  stats_idx = int(len(train_loader) / 30) # print stats ~30 times each epoch
  # NOTE: iS and iWS are two different integers uniquely assigned to the 
  # correct and incorrect speakers, respectively.
  for batch_idx, (v, f1, f2, target, iS, iWS) in enumerate(tqdm(train_loader)):
    v, f1, f2, target = v.to(device), f1.to(device), f2.to(device), target.to(device)
    assert len(v) == len(f1) and len(f1) == len(f2)
    optimizer.zero_grad()
    output: torch.tensor = model(v, f1, f2)

    loss = loss_f(output, target)
    loss.backward()
    optimizer.step()

    pred = output.argmax(dim=1, keepdim=True) # index of the max probability
    correct = pred.eq(target.view_as(pred)).flatten()
    correct_cnt += correct.sum().item()
    update_marginal_stats(correct_mrg, correct, iS)
    update_marginal_stats(correct_mrg, correct, iWS)

    # print stats
    if batch_idx % stats_idx == 0:
      tqdm.write(f'Epoch: {epoch}\n\tLoss: {round(loss.item(), 6)}')
      # accuracies
      cur_iter = (batch_idx + 1) * len(v)
      for acc_func, acc_str in acc_funcs:
        acc = acc_func(correct_cnt, cur_iter, correct_mrg)
        tqdm.write('\t' + acc_str + f': {acc}')
    # check timeout
    if (timeout > 0 and ((time.time() - start_time) / 60) >= timeout):
      print("\n****----Timeout!.----****\n")
      return True
  return False
      

def test(model, device, test_loader, loss_f, acc_funcs: list):
  model.eval()
  loss = 0
  correct_cnt = 0
  correct_mrg = dict()

  with torch.no_grad():
    # NOTE: iS and iWS are two different integers uniquely assigned to the 
    # correct and incorrect speakers, respectively.
    for v, f1, f2, target, iS, iWS in tqdm(test_loader):
      v, f1, f2, target = v.to(device), f1.to(device), f2.to(device), target.to(device)
      assert len(v) == len(f1) and len(f1) == len(f2)
      output: torch.tensor = model(v, f1, f2)

      loss += loss_f(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True) # index of the max probability
      correct = pred.eq(target.view_as(pred)).flatten()
      correct_cnt += correct.sum().item()
      update_marginal_stats(correct_mrg, correct, iS)
      update_marginal_stats(correct_mrg, correct, iWS)

  # print stats
  loss/=len(test_loader.dataset)
  print(f'Average Loss: {round(loss.item(), 6)}')
  # accuracies
  for acc_func, acc_str in acc_funcs:
    acc = acc_func(correct_cnt, len(test_loader.dataset), correct_mrg)
    print(acc_str + f': {acc}')


def save_model(model, model_path: str):
  torch.save(model.state_dict(), model_path)
  print("\n****----Model Saved----****\n")


# Accuracies (passed three arguments: int, int, dict[int, tuple[int, int]])
def accuracy(correct: int, tests: int, arg3):
  return float(correct / tests)

def avg_marginal_acc(arg1, arg2, correct_mrg: dict):
  total = 0
  for correct, tests in correct_mrg.values():
    total += accuracy(correct, tests, None)
  return total / len(correct_mrg)


# Calculates the multiplicative factor of learning rate decay
def get_gamma(lr_init:float, lr_final: float, epochs: int):
  assert 0 < lr_final and lr_final < lr_init and epochs > 0
  return 1 if epochs == 1 else math.exp(math.log(lr_final/lr_init)/(epochs-1))


def main():
##########################################################################################################NOT FINISHED
  # Available models. 
  # Maps a model keyword to a 3-tuple containing, in order:
  # - The models's constructor.
  # - The loss function
  # - A list of 2-tuples containing, in order:
  #   - An accuracy function
  #   - A name given to the accuracy function (to be printed)
  model_dict = {
    'baseline': (
      SVHF_Net, 
      F.nll_loss, 
      [
        (accuracy, "Total Accuracy"), 
        (avg_marginal_acc, "Average Marginal Accuracy")
      ]
    ),
    'random': (
      RandomNet, 
      F.nll_loss, 
      [
        (accuracy, "Total Accuracy"), 
        (avg_marginal_acc, "Average Marginal Accuracy")
      ]
    ),
  }
######################################################################################################################

  parser = argparse.ArgumentParser(description = 'SVHF-Net')
  parser.add_argument('--model', type=str, default='baseline', metavar='STR',
                      choices=list(model_dict.keys()),
                      help='Model to be used. Default: "baseline"')
  parser.add_argument('--lr_init', type=float, default=1e-2, metavar='FLOAT', 
                      help='''(If train is true) Initial learning rate used in 
                        training. Must be greater than 0. Default: 1e-2''')
  parser.add_argument('--lr_final', type=float, default=1e-8, metavar='FLOAT', 
                      help='''(If train is true) Minimum learning rate used in 
                        training. Must be less than lr_init and greater than 0. 
                        Default: 1e-8''')
  parser.add_argument('--w_decay', type=float, default=5e-4, metavar='FLOAT', 
                      help='''Weight decay (L2 penalty) used in training. 
                        Default: 5e-4''')
  parser.add_argument('--train', type=str, default='y', metavar='STR', 
                      choices=('y', 'n'),
                      help='"y" if training, "n" if testing. Default: "y"')
  parser.add_argument('--batch_sz', type=int, default=16, metavar='INT', 
                      help='''Batch size. Used for both train and validation
                        datasets if training. Default: 16.''')
  parser.add_argument('--load', type=str, default='n', metavar='STR', 
                      choices=('y', 'n'),
                      help='''"y" if the model is loaded from a saved state, 
                        "n" otherwise. Default: "n"''')
  parser.add_argument('--model_path', type=str, default='./models/trained.model', 
                      metavar='STR', 
                      help='''Path to load the model (if load is 
                        True or train is False) and/or store it (if train is 
                        True). Default: "./models/trained.model"''')
  parser.add_argument('--epochs', type=int, default=10, metavar='INT', 
                      help='''(If train is true) Number of epochs to train and 
                        validate the model. Must be greater than 0. 
                        Default: 10''')
  parser.add_argument('--timeout', type=int, default=120, metavar='INT', 
                      help='''(If train is true) Timeout for training -in 
                        minutes-. If < 1, no timeout is assumed. Default: 120''')
  parser.add_argument('--data_path', type=str, default='./data', metavar='STR', 
                      help='''Path to the data folder. Default: "./data"''')
  parser.add_argument('--workers', type=int, default=1, metavar='INT', 
                      help='''Number of subprocesses used to load data. Must be
                        greater or equal to 0. If 0, data is loaded in the main
                        process. Default: 1''')
  args = parser.parse_args()

  model, loss, accs = model_dict[args.model]
  device = torch.device("cuda") # forced cuda device 
  if args.train == 'y':
    model = model(args.model_path if args.load == 'y' else None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.w_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=get_gamma(args.lr_init, args.lr_final, args.epochs))
    train_loader, valid_loader = load_data(True, args.batch_sz, args.workers, args.data_path)
    for epoch in range(args.epochs):
      timeout = train(model, device, train_loader, loss, accs, optimizer, epoch, args.timeout)
      if timeout:
        break
      test(model, device, valid_loader, loss, accs)
      scheduler.step()
    save_model(model, args.model_path)
  else:
    model = model(args.model_path).to(device)
    test_loader = load_data(False, args.batch_sz, args.workers, args.data_path)
    test(model, device, test_loader, loss, accs)


if __name__ == "__main__":
  main()