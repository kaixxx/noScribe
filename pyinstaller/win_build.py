import os
import sys
from subprocess import Popen

noScribe_version = '0.5'
clean_build = True
run_pyinstaller_non_cuda = True
run_pyinstaller_cuda = True
run_nsis_non_cuda = True
run_nsis_cuda = True

conda_env_noncuda = 'noScribe_0_4'
conda_env_cuda = 'noScribe_0_4_cuda'
pyinstaller_path = 'C:\\users\\kai\\anaconda3\\envs\\noscribe_0_4\\lib\\site-packages\\pyinstaller'
nsis_path = 'C:\\Program Files (x86)\\NSIS\\makensis.exe'

script_dir = os.path.abspath(os.path.dirname(__file__))
final_report = '\n######################################\nResults:\n#######################################\n'

##### PyInstaller #####

def get_pyinstaller_out_path(cuda=False):
    if cuda:
        return os.path.join(script_dir, 'dist', 'noScribe_cuda')
    else:
        return os.path.join(script_dir, 'dist', 'noScribe_noncuda')

def run_pyinstaller(cuda=False):
    global final_report 
    print('##############################################################')
    print('PyInstaller cuda' if cuda else 'PyInstaller non cuda')

    pyinstaller_out_path = get_pyinstaller_out_path(cuda)
    if cuda:
        pyinstaller_cmd = f'conda activate {conda_env_cuda} &&'
    else:
        pyinstaller_cmd = f'conda activate {conda_env_noncuda} &&'
        
    pyinstaller_cmd += f'python  "{pyinstaller_path}" --noconfirm "{os.path.join(script_dir, "noScribe_combined_win.spec")}" --distpath {pyinstaller_out_path}'
    if clean_build:
        pyinstaller_cmd += ' --clean'
    
    print(pyinstaller_cmd)
    proc = Popen(pyinstaller_cmd, shell=True, cwd=script_dir)
    proc.communicate()
    if proc.returncode != 0:
        final_report += 'PyInstaller failed.\n'
        final_report += 'Cmd: ' + pyinstaller_cmd
        print(final_report)
        quit(proc.returncode)
    else:
        final_report += 'PyInstaller build with cuda succeded\n' if cuda else 'PyInstaller build without cuda succeded\n'

##### NSIS Installer #####

def run_nsis(cuda=False):
    def format_version(version_string):
        """
        Formats a version string to ensure it has four segments
        by appending ".0" for any missing segments.
        """
        target_length = 4
        segments = version_string.split('.')
        # Calculate how many additional "0" segments need to be appended
        missing_segments = target_length - len(segments)
        
        if missing_segments > 0:
            # Append "0" for each missing segment
            segments.extend(['0'] * missing_segments)

        return '.'.join(segments)
   
    global final_report
    pyinstaller_out_path = get_pyinstaller_out_path(cuda)
    installer_name = 'noScribe_setup_' + noScribe_version.replace('.', '_')
    if cuda:
        installer_name += '_cuda'
    installer_name += '.exe'
    installer_name = os.path.join(script_dir, 'win_installer', installer_name)
        
    print('##############################################################')
    print('NISIS cuda' if cuda else 'NSIS non cuda')
    
    # prepare template
    with open(os.path.join(script_dir, 'nsis_template.txt'), 'r', encoding="utf-8") as nsis_templ_file:
        nsis_templ = nsis_templ_file.read()
    nsis_templ = nsis_templ.replace('#*version*#', format_version(noScribe_version))

    # Recursively generate NSIS commands for installation and uninstallation
    # of directories and files from the specified directory.
    
    base_directory = os.path.join(pyinstaller_out_path, 'noScribe')

    install_entries = '' # "Section \"Install\"\n"
    uninstall_entries = '' # "Section \"Uninstall\"\n"

    directories_created = []  # Track directories for uninstall

    # Process directories
    for root, dirs, files in sorted(os.walk(base_directory, topdown=True), key=lambda x: x[0]):
        relative_path = os.path.relpath(root, base_directory)
        if relative_path == ".":
            relative_path = ""  # NSIS SetOutPath doesn't need a dot for the base path
            install_entries += 'SetOutPath "$INSTDIR"\n'
        else:
            # relative_path += "\\"  # Append backslash for NSIS SetOutPath command
            install_entries += 'SetOutPath "$INSTDIR\\{}"\n'.format(relative_path.replace(os.sep, "\\"))

        if relative_path:
            directories_created.append(relative_path.replace(os.sep, "\\"))
        
        for filename in files:
            # Generate File command for each file
            filepath = os.path.join(root, filename).replace(os.sep, "\\")
            install_entries += 'File "{}"\n'.format(filepath)

            # Generate Delete command for each file for uninstallation
            uninstall_entries += 'Delete "$INSTDIR\\{}"\n'.format(os.path.join(relative_path, filename).replace(os.sep, "\\"))

    # Generate RMDir commands for uninstallation, in reverse order
    for directory in reversed(directories_created):
        uninstall_entries += 'RMDir "$INSTDIR\\{}"\n'.format(directory)

    # Prepare NSIS script
    nsis_templ = nsis_templ.replace('#*installer_name*#', installer_name)
    nsis_templ = nsis_templ.replace('#*install_entries*#', install_entries, 1)
    nsis_templ = nsis_templ.replace('#*uninstall_entries*#', uninstall_entries, 1)

    # print(nsis_templ)
    print('Writing nsis_tmp.nsi')
    with open(os.path.join(script_dir, 'nsis_tmp.nsi'), 'w', encoding="utf-8") as nsis_out:
        nsis_out.write(nsis_templ)

    nsis_cmd = '"' + nsis_path + '" /V4 "' + os.path.join(script_dir, 'nsis_tmp.nsi') + '"'

    proc = Popen(nsis_cmd, shell=True, cwd=os.path.join(script_dir, 'win_installer'))
    proc.communicate()    
    if proc.returncode != 0:
        final_report += 'NSIS commpiler failed.\n'
        final_report += 'Cmd: ' + nsis_cmd
        print(final_report)
        quit(proc.returncode)
    else:
        final_report += 'NSIS compiler with cuda succeded\n' if cuda else 'NSIS compiler without cuda succeded\n'

########################## Main ################################

if run_pyinstaller_non_cuda:
    run_pyinstaller(cuda=False)
if run_pyinstaller_cuda:
    run_pyinstaller(cuda=True)

if run_nsis_non_cuda:
    run_nsis(cuda=False)
if run_nsis_cuda:
    run_nsis(cuda=True)

print(final_report)