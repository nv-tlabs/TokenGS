# Contributing to TokenGS

Thank you for your interest in contributing to TokenGS! We welcome contributions from the community.

TokenGS is released under the **Apache License 2.0**. By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

## Developer Certificate of Origin (DCO)

TokenGS requires that all contributors sign off on commits submitted to our projects. The sign-off is a simple line at the end of the explanation for the patch, which certifies that you wrote it or otherwise have the right to pass it on as an open-source patch.

### The DCO Process

By contributing to TokenGS, you agree to the [Developer Certificate of Origin](https://developercertificate.org/). This certifies that your contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

### Sign Your Work

To sign off on a commit, simply use the `--signoff` or `-s` option when committing changes:

```bash
git commit -s -m "Add new feature"
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

The sign-off must include your real name and email address.

### Full DCO Text

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Contribution Workflow

1. **Fork the repository**: Create your own fork of the TokenGS repository.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/tokengs.git
   cd tokengs
   ```

3. **Create a branch**: Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**: Implement your changes following the project's coding standards.

5. **Test your changes**: Ensure your changes work as expected and don't break existing functionality.

6. **Commit with sign-off**: Commit your changes with the DCO sign-off:
   ```bash
   git commit -s -m "Description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Submit a pull request**: Open a pull request from your fork to the main TokenGS repository with a clear description of your changes.

## Code Review Process

- All submissions require review before being merged.
- We may ask you to make changes to your contribution.
- All commits must include the DCO sign-off.

## Code Standards

### License Headers

All source code files must include the appropriate license header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

For NVIDIA-authored code, you may use SPDX identifiers as shown above. For contributions that include third-party code, ensure proper attribution and license information is maintained.

### Third-Party Dependencies

If your contribution introduces new third-party dependencies:
- Ensure the dependency's license is compatible with Apache 2.0
- Document the dependency in `pyproject.toml`

## Questions?

If you have questions about contributing, please open an issue in the repository.

## References

- [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- [Developer Certificate of Origin](https://developercertificate.org/)
- [SPDX License Identifiers](https://spdx.dev/learn/handling-license-info/)

