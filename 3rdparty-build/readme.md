## tensorflow-pack / 3rd Party genpack Tool

Python based build tool to create CMSIS Packs from local or http sources.

### Syntax
```console
python3 genpack.py --path=<packpath> [--date_tag --version=1.2.3]
```

- path : Specify the path to that contains the pack description and assets. 
- date_tag : Append the date of today as ddmmyyyy to the version number. Helps nightly CI builds.
- version : Optionally override a version specified in config.yml

### Directory Structure of Pack Assets

| File/Folder   | Details  |
| ------------- |:-------------:|
| contributions/add/     | Files that are copied untouched to the pack |
| contributions/merge/   | Template Files that are merged with variables before copied to pack (only pdsc supported now) |
| config.yml             | The configuration |

### config.yml Example

```yaml  
name: flatbuffers
vendor: tensorflow
version: 0.0.1

pdsc: ./contributions/merge/tensorflow.flatbuffers.pdsc
add:  ./contributions/add
build: build
out: ./

remote_source: #https://github.com/google/flatbuffers/archive/master.zip
local_source: ../../../tensorflow/tensorflow/lite/micro/tools/make/downloads/flatbuffers
src_dest: ./src
```




## Ethos-U dependency

The build produces only flatbuffers, gemmlowp, kissfft and ruy. Together with
`tensorflow-lite-micro`, these five packs share the existing candidate version.
`ARM::CMSIS-Ethos-U@1.0.0` is consumed from Arm's published release, not built or
renumbered here. The test workflow verifies its SHA256 against
`.github/scripts/cmsis_ethos_u.json` before installation.
