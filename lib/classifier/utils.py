
def get_elite(checkpoints: list, verbose: bool = False) -> str:
    """Get the best checkpoint based on min loss.

    Parameters
    ----------
    checkpoints : list
        List of checkpoints.
    verbose : bool, optional
        Print the best checkpoint, by default False
        
    Returns
    -------
    str
        Best checkpoint.
    """
    try:
        checkpoints.remove('.gitkeep')
    except ValueError:
        pass
    elite = None
    for idx, (checkpoint) in enumerate(checkpoints):
        try:
            if checkpoint.endswith('.ckpt'):
                if idx == 0 or elite is None:
                    elite = checkpoint
                else:
                    if float('.'.join(checkpoint.split("_")[2].split(".")[:2]).split('-')[0]) < float('.'.join(elite.split("_")[2].split(".")[:2]).split('-')[0]):
                        elite = checkpoint
        except IndexError:
            pass
    if verbose:
        print(f'Found elite checkpoint: {elite}')
    return elite


def colorstr(options, string_args):
    """Usage:
    
    >>> args = ['Good', 'Morning']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args)} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=list(['Python']))} "
    ...    f"and {colorstr(options=['cyan'], string_args=list(['C++']))}\n")
    Parameters
    ----------
    options : [type]
        [description]
    string_args : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {'black':          '\033[30m',  # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m',  # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}
    res = []
    for substr in string_args:
        res.append(''.join(colors[x] for x in options) +
                   f'{substr}' + colors['end'])
    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)
