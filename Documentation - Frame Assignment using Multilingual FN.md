# Documentation - Frame Assignment using Multilingual FN

All the necessary files reside in the folder `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs` and the file that is run is `multilingual_frame_assignment.py`.

The `sbatch` script (`task-multilingual-frame-assignment.slurm`) used to assign frames to the lexical unit clusters:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=14-00:00:00
#SBATCH --output=multilingual_frame_assignment.stdout
#SBATCH --error=multilingual_frame_assignment.err
#SBATCH --job-name="frame assignment using multilingual FN"

module load gcc/6.3.0 openmpi/2.0.1 python/3.6.6
module load singularity
export SINGULARITY_BINDPATH="/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs:/mnt"

singularity exec production_multilingual4.sif python3 -u /mnt/multilingual_frame_assignment.py --file_model_result="/mnt/data/0102/lu_cluster_affinity_propagation.pkl" --folder_unseen_LUs="/mnt/data/0102" --folder_multilingual_frame_assignment="/mnt/data/0102" > "./data/0102/multilingual_frame_assignment_output.out"
```

The argument `--file_model_result` specifies the folder that stores the `lu_cluster_affinity_propagation.pkl` result (from the Clustering Lexical Units Documentation).

The argument `--folder_unseen_LUs` specifies the folder that stores the pickle files `unseen_lus-*.seg.p`. 

The argument `--folder_multilingual_frame_assignment` specifies the folder to 
store `clusters_to_multilingual_potential_frames_counters.p`, which the result file of a dictionary that maps cluster indexes to a collections.Counter dictionary of potential frames retrieved from multilingual framenets. 



**Output**

`clusters_to_multilingual_potential_frames_counters.p`

It is a pickled file of a dictionary that maps cluster indexes to a collections.Counter dictionary of potential frames retrieved from multilingual framenets.

For example: 

```
{..., 22: Counter({'People_by_age': 1, 'frm_people_by_age': 1}), ...}
```



---

## Implementation Details

1. **Korean FN frame assignment**

The reason for choosing Korean FN is that the library and APIs are fully implemented and it is available on GitHub - https://github.com/machinereading/koreanframenet. 

In the code below, the new lexical units in a cluster are translated into korean, and they are parsed through the Korean FN APIs to search for relevant frames. Notice that the code below doesn't immediately select the best frame for the particular cluster. Instead, I used `collections.Counter()` dictionary to track the frames and their respective counts that are associated with the lexical units. The reason is that such implementation makes integration with other FrameNet easier as we choose the best frame representation at the end of the pipeline.

```python
def koreanFN_frame_assignment(clusters_to_LU_tensor_tuples, clusters_to_potential_frames_counters):
    """
    :param clusters_to_LU_tensor_tuples:
    # {0: [('rob.n', array([ 0.11830516, -0.53951496, -0.07126138, ...,  0.25920865, 0.7204548 , -1.4147129 ], dtype=float32)),
    #      ('nick.n', array([-0.8950612 , -0.2902073 ,  0.07615741, ...,  0.16403697, -0.00709647, -0.0444046 ], dtype=float32)),
    #      ('ligh.n', array([ 0.28627014, -0.28651422,  0.26550704, ..., -0.51892674, -0.27826402,  0.21954602], dtype=float32)),
    #      ('ralphy.n', array([-1.2499862 , -0.18947789, -0.18876693, ...,  0.8362526 , 0.93789047,  0.29156995], dtype=float32))],
    #  6: [('carloss.n', array([ 0.68604803,  1.0085262 ,  0.74925   , ..., -0.38497278, 0.7514505 ,  0.20123959], dtype=float32)),
    #      ('delvante.n', array([-0.17765829, -1.2151165 ,  0.81548035, ..., -0.700882, 0.17137729,  0.3607476 ], dtype=float32)),
    #      ... ], ...}
    :param clusters_to_potential_frames_counters: dictionary that maps the cluster index to the inbuilt Counter dictionary which stores
                                                  the frequency of potential frames
    :return: None

    :Function: Modify the dictionary clusters_to_potential_frames_counters
    {cluster_idx : {frm_1: 1, frm_2: 3, ...}, ...}
    """
    translator = Translator(to_lang="ko")
    for cluster_idx, LU_tensor_tuples in clusters_to_LU_tensor_tuples.items():
        print("Process Cluster Index:", cluster_idx)
        potential_frames_counter = collections.Counter()
        for LU_name, tensor in LU_tensor_tuples:
            try:
                lemma = ''.join(LU_name.split(".")[:-1])
                korean_translated_lemma = translator.translate(lemma)
                while korean_translated_lemma.startswith("MYMEMORY WARNING"):
                    print("Sleep Starts..")
                    time.sleep(86400)  # API request limit reached. sleep for 24hours
                    korean_translated_lemma = translator.translate(lemma)
                    print("Sleep Ends..")
                korean_lus = kfn.lus_by_word(korean_translated_lemma)
                for korean_lu in korean_lus:
                    # korean_lu = {'lu': '입증하다.v', 'frame': 'Statement', 'lu_id': 5565}
                    potential_frames_counter[korean_lu['frame']] += 1
            except Exception as e:
                print(str(e))
                continue

        if cluster_idx in clusters_to_potential_frames_counters:
            for frame, count in potential_frames_counter.items():
                clusters_to_potential_frames_counters[cluster_idx][frame] += count
        else:
            clusters_to_potential_frames_counters[cluster_idx] = potential_frames_counter
```



2. **Brasil FN frame assignment**

The reason for choosing Korean FN is that the library and APIs are fully implemented and it is available on GitHub - https://github.com/machinereading/koreanframenet. 

In the code below, the new lexical units in a cluster and the exemplar sentences are translated into portugese, and they are parsed through the BrasilFN APIs to search for relevant frames for all the lexical units (including new or existing lexical units) in the exemplar sentences. Then, a for-loop is used to find if the new lexical units are assigned a frame from BrasilFN. 

Notice that the code below doesn't immediately select the best frame for the particular cluster. Instead, I used `collections.Counter()` dictionary to track the frames and their respective counts that are associated with the lexical units. The reason is that such implementation makes integration with other FrameNet easier as we choose the best frame representation at the end of the pipeline.

```python
def BrasilFN_frame_assignment(clusters_to_LU_tensor_tuples, clusters_to_potential_frames_counters, unseen_lus_file):
    pt_translator = Translator(to_lang="pt")
    en_translator = Translator(to_lang="en")
    data = fnbr.Data()
    H = pickle.load(open(unseen_lus_file, 'rb'))
    for cluster_idx, LU_tensor_tuples in clusters_to_LU_tensor_tuples.items():
        print("Process Cluster Index:", cluster_idx)
        potential_frames_counter = collections.Counter()
        for LU_name, tensor in LU_tensor_tuples:
            sentences = H[LU_name]
            for sent, idx in sentences:
                try:
                    portugese_sent = pt_translator.translate(sent)
                    while portugese_sent.startswith("MYMEMORY WARNING"):
                        print("Sleep Starts..")
                        time.sleep(86400)  # API request limit reached. sleep for 24hours
                        print("Sleep Ends..")
                        portugese_sent = pt_translator.translate(lemma)
                    print(sent, ":", portugese_sent)
                    data.sentence = portugese_sent
                    dis = fnbr.DisambiguationService()
                    dis.data = data
                    res = dis.sentence()
                    if res:
                        lemma = ''.join(LU_name.split(".")[:-1])
                        lemma_translated = pt_translator.translate(lemma)
                        while lemma_translated.startswith("MYMEMORY WARNING"):
                            print("Sleep Starts..")
                            time.sleep(86400)  # API request limit reached. sleep for 24hours
                            print("Sleep Ends..")
                            lemma_translated = pt_translator.translate(lemma)

                        for brasil_lemma, frame in res['frames'].items():
                            brasil_lemma_translated = en_translator.translate(brasil_lemma)
                            while brasil_lemma_translated.startswith("MYMEMORY WARNING"):
                                print("Sleep Starts..")
                                time.sleep(86400)  # API request limit reached. sleep for 24hours
                                print("Sleep Ends..")
                                brasil_lemma_translated = pt_translator.translate(lemma)
                            if lemma_translated == brasil_lemma or brasil_lemma_translated == lemma:
                                potential_frames_counter[frame] += 1

                except Exception as e:
                    print("Error:", str(e), ":", sent)

        if cluster_idx in clusters_to_potential_frames_counters:
            for frame, count in potential_frames_counter.items():
                clusters_to_potential_frames_counters[cluster_idx][frame] += count
        else:
            clusters_to_potential_frames_counters[cluster_idx] = potential_frames_counter


    return clusters_to_potential_frames_counters

```

#### Future Frame Assignment to Clusters of New Lexical Units

There are a few future improvements for assigning the best frame for the cluster using multilingual framenets. 

1. Use more multilingual framenets such as SwedenFN and GermanFN. Some of the framenets such as Japanese FN and Spanish FN do not have API and dataset publicly available, but their web interface for online visualization is available. In the future, when there are more APIs released for these multilingual framenets, they can be incorporated into the pipeline.

2. Merging similar frames are necessary because of different naming conventions for different FrameNet. For example, "People_by_age" in KoreanFN and "frm_people_by_age" in BrasilFN are both the same frame.

3. Currently, each cluster is associated to a dictionary of potential frames, mapped to their frequency of occurrences. If the occurrences of a particularly frame is more than other frames, it would be considered the best frame for the cluster. 

   The reasoning is that if a cluster hypothetically represents a frame, then most of the lexical units in the cluster share the same frame; therefore, the frequency of the retrieved frame that best represent the cluster will be high. 

   However, this could be problematic because first, the lexical units could be polysemous and the concept of one-frame-for-one-cluster may not be true. Second, if there's an even spread of frequency, this rule-of-thumb of choosing the frame with maximum frequency will not work.



---

## Documentation of Singularity Containers

I use VagrantBox in MacOS to create the Singularity container `/home/zxy485/zxy485gallinahome/week9-12/unseen_LUs/production.sif` for this task of assigning frames to lexical unit clusters using KoreanFN and BrasilFN. 

**Dependencies Installation in Singularity Container**

```bash
pip3 --no-cache-dir install translate
sudo apt-get install python3.6-dev libmysqlclient-dev
pip3 install --no-binary mysqlclient mysqlclient

# see thread https://webkul.com/blog/setup-locale-python3/ , https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue
apt-get install -y locales
locale-gen "en_US.UTF-8"
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
```



---

## Documentation of Bugs / Challenges
**ImportError related to libmysqlclient** - This error occurs when I used PyDaisy to retrieve BrasilFN frames from the MySQL server. It is due to the incorrect installation of the library `mysqlclient`. In fact, it is a common error when we directly `pip install mysqlclient` in the container.

```bash
Traceback (most recent call last):
  File "/mnt/multilingual_frame_assignment.py", line 12, in <module>
    import py_daisy.DisambiguationService as fnbr
  File "/mnt/py_daisy/DisambiguationService.py", line 3, in <module>
    import py_daisy.Frame as Frame
  File "/mnt/py_daisy/Frame.py", line 1, in <module>
    from py_daisy.Database import Database
  File "/mnt/py_daisy/Database.py", line 1, in <module>
    import MySQLdb
  File "/home/zxy485/.local/lib/python3.6/site-packages/MySQLdb/__init__.py", line 18, in <module>
    from . import _mysql
ImportError: libmysqlclient.so.18: cannot open shared object file: No such file or directory
```

The solution is to pip install by not using binary packages – `pip3 install --no-binary mysqlclient mysqlclient`. The [`no-binary`](https://pip.pypa.io/en/stable/reference/pip_install/#install-no-binary) option is so that pip builds it fresh and links to the correct library. Note that `mysqlclient` needs to be mentioned twice. The first occurrence is the name of the package to apply the `no-binary` option to, the second specifies the package to install.



**UnicodeDecodeError** - This error occurs when I used `koreanframenet` library to retrieve KoreanFN frames for the korean lexical units. It is due to Python reading the lexical units in ASCII encoding, when it should be UTF-8 encoding, owing to the Singularity container's Ubuntu environment. 

When I run `$ singularity exec production_multilingual4.sif locale` I received error messages that states that LC_CTYPE, LC_MESSAGES and LC_ALL cannot be set to default locale, which is supposed to be en_US.UTF-8.

```bash
# UnicodeDecodeError
Traceback (most recent call last):
  File "/mnt/try.py", line 2, in <module>
    kfn = koreanframenet.interface(version=1.1)
  File "/mnt/koreanframenet/koreanframenet.py", line 44, in __init__
    self.kfn_lus = json.load(f)
  File "/usr/lib/python3.6/json/__init__.py", line 296, in load
    return loads(fp.read(),
  File "/usr/lib/python3.6/encodings/ascii.py", line 26, in decode
    return codecs.ascii_decode(input, self.errors)[0]
UnicodeDecodeError: 'ascii' codec can't decode byte 0xec in position 31: ordinal not in range(128)

# and also
[zxy485@hpc4 unseen_LUs]$ singularity exec production_multilingual4.sif locale
WARNING: seccomp requested but not enabled, seccomp library is missing or too old
locale: Cannot set LC_CTYPE to default locale: No such file or directory
locale: Cannot set LC_MESSAGES to default locale: No such file or directory
locale: Cannot set LC_ALL to default locale: No such file or directory
LANG=en_US.UTF-8
LANGUAGE=
LC_CTYPE=en_US.UTF-8
LC_NUMERIC="en_US.UTF-8"
LC_TIME="en_US.UTF-8"
LC_COLLATE="en_US.UTF-8"
LC_MONETARY="en_US.UTF-8"
LC_MESSAGES="en_US.UTF-8"
LC_PAPER="en_US.UTF-8"
LC_NAME="en_US.UTF-8"
LC_ADDRESS="en_US.UTF-8"
LC_TELEPHONE="en_US.UTF-8"
LC_MEASUREMENT="en_US.UTF-8"
LC_IDENTIFICATION="en_US.UTF-8"
LC_ALL=
```

The solution is to fix the Python 3 POSIX locale database and functionality, which is running the following lines when building the Singularity container.

```bash
apt-get install -y locales
locale-gen "en_US.UTF-8"
export LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
```



**List index out of range error** - This error occurs when I used PyDaisy to retrieve BrasilFN frames for the new lexical units. I believe the error is due to there's none of BrasilFN frames are associated to any of the words in the exemplar sentences of the lexical units. An example of such sentence (in Portugese) is "o açougue", which is translated from NewsScape sentence "the butcher shop."

There's no fix or further actions required to tackle the error because the issue involves expansion of BrasilFN. Therefore, the error is handle by try-except statement, and there's 



**MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY. NEXT AVAILABLE IN  12 HOURS 28 MINUTES 55 SECONDSVISIT HTTPS://MYMEMORY.TRANSLATED.NET/DOC/USAGELIMITS.PHP TO TRANSLATE MORE** - This error occurs due to the API translation limit when I use the `translate` Python library. Free, anonymous usage is limited to 1000 words/day.

There are two adjustments made. The first adjustment is made to code by removing the translation for the entire sentences in the KoreanFN frame assignment function. The translation to korean lexical units are done by direct translation, with the trade off that the translation might not be accuracy due to polysemous words. The second adjustment is to use `sleep` (for 24 hours) whenever this error is encountered.
