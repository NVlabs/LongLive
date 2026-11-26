# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity store: applies policy updates, keeps evidence/provenance, and decides what to redraw."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..story import StorySetup
from .entity import Entity, Fact, ImageVersion
from .snapshot import Reference
from .update import FactUpdate, StateUpdate, is_transient_attr, slug


@dataclass
class ApplyResult:
    created: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)  # entities whose drawn (visual) state changed
    revoked: List[Tuple[str, str]] = field(default_factory=list)  # (entity id, attribute)
    ignored: List[Tuple[str, str]] = field(default_factory=list)  # updates blocked when revision is off
    # revokes of an attribute that the same update also writes: the new value replaces the old one, it is kept
    superseded: List[Tuple[str, str]] = field(default_factory=list)


class EntityStore:
    """Thread-safe store of entities.

    ``split``: draw only persistent facts (Method 2, C1).
    ``revision``: memory is the current belief; newer evidence overrides older, and the policy may
    withdraw earlier (usually inferred) facts. With ``revision=False`` an inferred fact can never
    override an existing one and withdrawals are ignored (ablation).
    """

    def __init__(self, split: bool = True, revision: bool = True):
        self.split = split
        self.revision = revision
        self.entities: Dict[str, Entity] = {}
        self.scene: Optional[str] = None
        self.active: List[str] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ setup
    def seed_cast(self, setup: StorySetup, split: Optional[Dict[str, Dict]] = None) -> List[str]:
        """Protagonists from the cast cards. ``split`` (memory.cast_split, see MemoryAgent.bootstrap) maps a cast id to
        {"identity": str, "facts": {attr: value}}: the description keeps only what never changes and the card's
        changeable appearance (clothing, facial hair, accessories, ...) becomes explicit persistent facts, so a later
        prompt can update it like any other state. Without a split the whole card is the (immutable) description."""
        with self._lock:
            ids = []
            for member in setup.cast:
                eid = slug(member.id)
                part = (split or {}).get(eid) or (split or {}).get(member.id)
                entity = Entity(id=eid, kind="character", name=member.name,
                                description=part["identity"] if part else member.description, protagonist=True)
                if part:
                    for attr, value in part["facts"].items():
                        entity.facts[attr] = Fact(attr=attr, value=value, persistent=True, source="explicit", shot=-1)
                    entity.history.append({"shot": -1, "event": "cast_split", "card": member.description})
                self.entities[eid] = entity
                ids.append(eid)
            self.active = list(ids)
            return ids

    # ---------------------------------------------------------------- updates
    def apply(self, update: StateUpdate, shot: int) -> ApplyResult:
        result = ApplyResult()
        with self._lock:
            before = {eid: e.signature(self.split) for eid, e in self.entities.items()}
            for eu in update.entities:
                entity = self.entities.get(eu.id)
                if entity is None:
                    entity = Entity(id=eu.id, kind=eu.kind, name=eu.name or eu.id,
                                    description=eu.description or eu.name or eu.id, created_shot=shot)
                    self.entities[eu.id] = entity
                    result.created.append(eu.id)
                elif eu.description and not entity.description:
                    entity.description = eu.description
                if eu.present:
                    entity.last_seen_shot = shot
                    entity.seen_count += 1
                    entity.status = "active"
                for fact in eu.facts:
                    self._set_fact(entity, fact, shot, result)
                written = {fact.attr for fact in eu.facts}
                for attr in eu.revoke:
                    if attr in written:
                        # "revoke the old value and write a new one": the new value stands (it was just written)
                        result.superseded.append((entity.id, attr))
                        continue
                    if attr not in entity.facts:
                        continue
                    if not self.revision:
                        result.ignored.append((entity.id, attr))
                        continue
                    old = entity.facts.pop(attr)
                    entity.history.append({"shot": shot, "attr": attr, "old": old.value, "event": "revoked"})
                    result.revoked.append((entity.id, attr))
            if update.scene:
                if update.scene not in self.entities:
                    self.entities[update.scene] = Entity(id=update.scene, kind="scene", name=update.scene,
                                                         description=update.scene.replace("_", " "), created_shot=shot)
                    result.created.append(update.scene)
                self.scene = update.scene
                self.entities[update.scene].last_seen_shot = shot
                self.entities[update.scene].status = "active"
            if update.active:
                self.active = [a for a in dict.fromkeys(update.active) if a in self.entities]
            for eid in update.retire:
                if eid in self.entities and not self.entities[eid].protagonist:
                    self.entities[eid].status = "dormant"
                    self.active = [a for a in self.active if a != eid]
            for eid, entity in self.entities.items():
                if before.get(eid) != entity.signature(self.split) and eid not in result.created:
                    result.changed.append(eid)
        return result

    def _set_fact(self, entity: Entity, fact: FactUpdate, shot: int, result: ApplyResult) -> None:
        if fact.persistent and entity.kind == "character" and is_transient_attr(fact.attr):
            fact = FactUpdate(fact.attr, fact.value, False, fact.source)
        old = entity.facts.get(fact.attr)
        if old is not None and old.value == fact.value:
            if fact.source == "explicit" and old.source == "inferred":
                old.source, old.shot = "explicit", shot  # inference confirmed by the prompt
            old.persistent = fact.persistent
            return
        if old is not None and not self.revision and fact.source == "inferred":
            result.ignored.append((entity.id, fact.attr))
            return
        if old is not None:
            entity.history.append({"shot": shot, "attr": fact.attr, "old": old.value, "new": fact.value,
                                   "old_source": old.source, "event": "updated"})
        entity.facts[fact.attr] = Fact(attr=fact.attr, value=fact.value, persistent=fact.persistent,
                                       source=fact.source, shot=shot)

    # ----------------------------------------------------------------- images
    def add_image(self, entity_id: str, image: ImageVersion) -> None:
        with self._lock:
            self.entities[entity_id].images.append(image)

    def needs_image(self) -> List[str]:
        with self._lock:
            return [eid for eid, e in self.entities.items()
                    if (e.status == "active" or e.protagonist) and e.needs_image(self.split)]

    # ------------------------------------------------------------------ views
    def get(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def table(self) -> List[Dict]:
        with self._lock:
            return [e.table_row() for e in self.entities.values()]

    def protagonists(self) -> List[str]:
        return [eid for eid, e in self.entities.items() if e.protagonist]

    def references(self, members: List[str], max_images: int, multi_view: bool = True) -> List[Reference]:
        """Current entity images for reference-injection backends; an entity's views are never split."""
        refs: List[Reference] = []
        with self._lock:
            for eid in dict.fromkeys(members):
                entity = self.entities.get(eid)
                if entity is None or entity.latest is None:
                    continue
                views = list(entity.latest.views.items())
                views = views if multi_view else views[:1]
                if len(refs) + len(views) > max_images:
                    views = views[:max(0, max_images - len(refs))]
                refs.extend(Reference(eid, entity.name, entity.kind, view, path, entity.describe(self.split))
                            for view, path in views)
                if len(refs) >= max_images:
                    break
        return refs

    def to_json(self) -> Dict:
        with self._lock:
            return {"scene": self.scene, "active": list(self.active),
                    "entities": {eid: e.to_json() for eid, e in self.entities.items()}}

    @classmethod
    def load(cls, path: Path, split: bool = True, revision: bool = True) -> "EntityStore":
        data = json.loads(Path(path).read_text())
        store = cls(split=split, revision=revision)
        store.scene = data.get("scene")
        store.active = list(data.get("active") or [])
        store.entities = {eid: Entity.from_json(e) for eid, e in (data.get("entities") or {}).items()}
        return store

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False))
