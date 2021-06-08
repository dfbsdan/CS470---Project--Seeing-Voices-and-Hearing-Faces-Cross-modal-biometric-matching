import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data_loader import load_data
from models import SVHF_Net
import time


def train(model, device, train_loader, loss_f, acc_funcs: list, optimizer, 
          epoch: int, timeout: int):
  model.train()
  correct = 0
  total_iter = len(train_loader.dataset)
  start_time = time.time()
  # NOTE: iS and iWS are two different integers uniquely assigned to the 
  # correct and incorrect speakers, respectively.
  for batch_idx, (v, f1, f2, target, iS, iWS) in enumerate(train_loader):
    v, f1, f2, target = v.to(device), f1.to(device), f2.to(device), target.to(device)
    assert len(v) == len(f1) and len(f1) == len(f2)
    optimizer.zero_grad()
    output = model(v, f1, f2)

    loss = loss_f(output, target)
    loss.backward()
    optimizer.step()

    pred = output.argmax(dim = 1, keepdim = True) # index of the max probability
    correct += pred.eq(target.view_as(pred)).sum().item()

    # print stats
    if batch_idx % 10 == 0:
      iter = batch_idx * len(v)
      percentage = int(100. * batch_idx / len(train_loader))
      print(f'Train epoch: {epoch} [{iter}/{total_iter} ({percentage}%)]')
      print(f'\tLoss: {round(loss.item(), 6)}')
##########################################################################################################NOT FINISHED
      # accuracies
      for acc_func, acc_str in acc_funcs:
        acc = acc_func(_)
        print('\t' + acc_str + f': {acc}\n')
######################################################################################################################

    if (timeout > 0 and ((time.time() - start_time) / 60) >= timeout):
      return True
  return False
      

def test(model, device, test_loader, loss_f, acc_funcs: list):
  model.eval()
  loss = 0
  correct = 0

  with torch.no_grad():
    # NOTE: iS and iWS are two different integers uniquely assigned to the 
    # correct and incorrect speakers, respectively.
    for v, f1, f2, target, iS, iWS in test_loader:
      v, f1, f2, target = v.to(device), f1.to(device), f2.to(device), target.to(device)
      assert len(v) == len(f1) and len(f1) == len(f2)
      output = model(v, f1, f2)

      loss += loss_f(output, target).item()
      pred = output.argmax(dim = 1, keepdim = True) # index of the max probability
      correct += pred.eq(target.view_as(pred)).sum().item()
  # print stats
  loss/=len(test_loader.dataset)
  print(f'Average Loss: {round(loss.item(), 6)}')
  percentage = 100. * correct / len(test_loader.dataset)
  print(f'Correct tests: {correct}/{len(test_loader.dataset)} ({percentage}%)')
##########################################################################################################NOT FINISHED
  # accuracies
  for acc_func, acc_str in acc_funcs:
    acc = acc_func(_)
    print(acc_str + f': {acc}\n')
######################################################################################################################


def save_model(model, file_name: str):
  torch.save(model.state_dict(), "models/" + file_name)
  print("\n****----Model Saved----****\n")


##########################################################################################################NOT FINISHED
# Accuracies

def accuracy(correct: int, tests: int):
  return correct / test_cnt

def avg_marginal_acc():
  return 
######################################################################################################################


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
    'baseline': (SVHF_Net, F.binary_cross_entropy, 
      [
        (accuracy, "Total Accuracy"), 
        (avg_marginal_acc, "Average Marginal Accuracy")
      ]),
  }
######################################################################################################################

  parser = argparse.ArgumentParser(description = 'SVHF-Net')
  parser.add_argument('--model', type=str, default='baseline', metavar='STR',
                      choices=list(model_dict.keys()),
                      help='Model to be used. Default: "baseline"')
  parser.add_argument('--train', type=bool, default=True, metavar='BOOL', 
                      help='True if training, False if testing. Default: True')
  parser.add_argument('--batch_sz', type=int, default=16, metavar='INT', 
                      help='''Batch size. Used for both train and validation
                        datasets if training. Default: 16.''')
  parser.add_argument('--load', type=bool, default=False, metavar = 'BOOL', 
                      help='''True if the model is loaded from a saved state, 
                        False otherwise. Default: False''')
  parser.add_argument('--file', type=str, default='trained.model', metavar='STR', 
                      help='''Name of the file to load the model (if load is 
                        True or train is False) and/or store it (if train is 
                        True). Default: "trained.model"''')
  parser.add_argument('--epochs', type=int, default=100, metavar='INT', 
                      help='''(If train is true) Number of epochs to train and 
                        validate the model. Default: 100''')
  parser.add_argument('--timeout', type=int, default=120, metavar='INT', 
                      help='''(If train is true) Timeout for training -in 
                        minutes-. If < 1, no timeout is assumed. Default: 120''')
  args = parser.parse_args()

  model, loss, accs = model_dict[args.model]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if args.train:
    model = model(args.file if args.load else None).to(device)
##########################################################################################################NOT FINISHED
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)
######################################################################################################################
    train_loader, valid_loader = load_data(True, args.batch_sz)
    for epoch in range(args.epochs):
      timeout = train(model, device, train_loader, loss, accs, optimizer, epoch, args.timeout)
      test(model, device, valid_loader, loss, accs)
      if timeout:
        print("\n****----Timeout! Saving model...----****\n")
        break
    save_model(model, args.file)
  else:
    model = model(args.file).to(device)
    test_loader = load_data(False, args.batch_sz)
    test(model, device, test_loader, loss, accs)


if __name__ == "__main__":
  main()